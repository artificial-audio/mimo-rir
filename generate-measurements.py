"""Generate and read back periodic MIMO sweeps, preserving every repetition.

Adapted from the four-condition AudioLab generator used on 2026-10-23.
New files add 100 ms digital silence at each file boundary.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import irfft, rfft

from mimo_rir import periodic_sweep


def make_metadata(*, waveform='exponential', mode='simultaneous', speakers=128,
                  fs=48000, response_seconds=1., repetitions=3, gain=.5, f1=20.):
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
           for v in (speakers, fs, repetitions)):
        raise ValueError('speakers, fs and repetitions must be positive integers')
    if not np.isfinite(response_seconds) or response_seconds <= 0:
        raise ValueError('response_seconds must be positive and finite')
    if not np.isfinite(gain) or not 0 < gain <= 1:
        raise ValueError('gain must be in (0, 1]')
    if waveform not in ('linear', 'exponential') or mode not in ('single-speaker', 'simultaneous'):
        raise ValueError('unknown waveform or schedule')
    if not np.isfinite(f1) or not 0 < f1 < fs / 2:
        raise ValueError('f1 must be between zero and Nyquist')
    length = 2 * round(response_seconds * fs / 2)
    if length < 2:
        raise ValueError('response support must span at least two samples')
    period = (2 if mode == 'single-speaker' else speakers) * length
    fade, padding = max(2, round(.01 * fs)), max(1, round(.1 * fs))
    pre = length + fade
    block = pre + repetitions * period + fade + length
    blocks = speakers if mode == 'single-speaker' else 1
    return dict(schema_version=1, sample_rate=fs, outputs=speakers,
                waveform=waveform, mode=mode, repetitions=repetitions,
                response_samples=length, period_samples=period, pre_roll_samples=pre,
                fade_samples=fade, block_samples=block, blocks=blocks,
                leading_silence_samples=padding, trailing_silence_samples=padding,
                frames=2 * padding + blocks * block, gain=gain,
                generator=dict(speakers=period // length, ir_len=length,
                               fs=fs, waveform=waveform, f1=f1),
                measured_starts_samples=[[padding + j * block + pre + p * period
                                          for p in range(repetitions)] for j in range(blocks)])


def samples(meta, base, a, b):
    """Playback samples [a,b), with internal wraps and repetitions left intact."""
    outputs = meta['outputs']
    t = np.arange(a, b) - meta['leading_silence_samples']
    block, length, fade = meta['block_samples'], meta['response_samples'], meta['fade_samples']
    relative = t % block
    live = block - length
    envelope = (1 - np.cos(np.pi * np.clip(relative / (fade - 1), 0, 1))) / 2
    envelope *= (1 - np.cos(np.pi * np.clip((live - 1 - relative) / (fade - 1), 0, 1))) / 2
    envelope[(relative >= live) | (t < 0) | (t >= meta['blocks'] * block)] = 0
    phase = relative - meta['pre_roll_samples']
    if meta['mode'] == 'simultaneous':
        return meta['gain'] * envelope[:, None] * base[(phase[:, None] - length * np.arange(outputs)) % len(base)]
    result = np.zeros((b - a, outputs))
    active = (t >= 0) & (t < meta['blocks'] * block)
    indices = np.flatnonzero(active)
    result[indices, t[active] // block] = meta['gain'] * envelope[active] * base[phase[active] % len(base)]
    return result


def verify(path, meta, base):
    """Read every exported sample; check quantization, routing and repeated periods."""
    hashes = [[hashlib.sha256() for _ in row] for row in meta['measured_starts_samples']]
    n, fs = meta['period_samples'], meta['sample_rate']
    reference = np.empty(n)
    peak = error = 0.
    with sf.SoundFile(str(path)) as src:
        if (src.samplerate, src.channels, len(src), src.subtype) != (fs, meta['outputs'], meta['frames'], 'PCM_24'):
            raise ValueError('exported WAV header does not match its schedule')
        for a in range(0, meta['frames'], fs):
            b = min(a + fs, meta['frames'])
            integers = src.read(b - a, dtype='int32', always_2d=True)
            actual = integers.astype(float) / 2**31
            expected = samples(meta, base, a, b)
            peak = max(peak, float(abs(actual).max()))
            error = max(error, float(abs(actual - expected).max()))
            if meta['mode'] == 'single-speaker' and np.any(np.count_nonzero(integers, axis=1) > 1):
                raise ValueError('overlapping active output channels')
            for j, starts in enumerate(meta['measured_starts_samples']):
                for p, start in enumerate(starts):
                    lo, hi = max(a, start), min(b, start + n)
                    if hi > lo:
                        hashes[j][p].update(integers[lo - a:hi - a].tobytes())
                        if j == 0 and p == 0:
                            reference[lo - start:hi - start] = actual[lo - a:hi - a, 0]
    digests = [[h.hexdigest() for h in row] for row in hashes]
    if error >= 1.21e-7 or peak > meta['gain'] + 1.21e-7 or any(len(set(row)) != 1 for row in digests):
        raise ValueError('exported waveform failed quantization, level or repetition check')
    spectrum = rfft(reference)
    if np.any(abs(spectrum) == 0):
        raise ValueError('quantized reference has a zero spectral bin')
    truth = np.zeros(n)
    for m in range(meta['outputs'] if meta['mode'] == 'simultaneous' else 1):
        truth[m * meta['response_samples']] = .3
        truth[(m + 1) * meta['response_samples'] - 1] = .1
    recovered = irfft(rfft(meta['gain'] * base) * rfft(truth) / spectrum, n)
    inverse_error = float(np.linalg.norm(recovered - truth) / np.linalg.norm(truth))
    if inverse_error >= 1e-4:
        raise ValueError('quantized-reference inverse check failed')
    return dict(all_samples_checked=True, max_quantization_error=error, peak=peak,
                period_sha256=digests, spectrum_condition=float(abs(spectrum).max() / abs(spectrum).min()),
                simulation_relative_error=inverse_error)


def generate(out, **options):
    out = Path(out)
    targets = [out.with_name(out.name + ext) for ext in ('.wav', '.json', '-verification.json')]
    if any(p.exists() for p in targets):
        raise FileExistsError('Output WAV, schedule or verification already exists')
    meta = make_metadata(**options)
    base = periodic_sweep(meta['generator'])
    out.parent.mkdir(parents=True, exist_ok=True)
    wav, schedule, report = targets
    print(f'Writing {wav}: {meta["frames"]/meta["sample_rate"]:.3f} s, {meta["outputs"]} channels', flush=True)
    with sf.SoundFile(str(wav), 'x', meta['sample_rate'], meta['outputs'], subtype='PCM_24', format='RF64') as dest:
        for a in range(0, meta['frames'], meta['sample_rate']):
            dest.write(samples(meta, base, a, min(a + meta['sample_rate'], meta['frames'])))
    schedule.write_text(json.dumps(meta, indent=2) + '\n')
    report.write_text(json.dumps(verify(wav, meta, base), indent=2) + '\n')
    print(f'PASS: {schedule}; {report}', flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--waveform', choices=['linear', 'exponential', 'both'], default='exponential')
    parser.add_argument('--mode', choices=['single-speaker', 'simultaneous', 'both'], default='simultaneous')
    parser.add_argument('--speakers', type=int, default=128)
    parser.add_argument('--fs', type=int, default=48000)
    parser.add_argument('--response-seconds', type=float, default=1.)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--gain', type=float, default=.5)
    parser.add_argument('--f1', type=float, default=20.)
    args = parser.parse_args()
    for wave in ('linear', 'exponential') if args.waveform == 'both' else (args.waveform,):
        for mode in ('single-speaker', 'simultaneous') if args.mode == 'both' else (args.mode,):
            generate(args.output_dir / f'{wave}-{mode}-{args.speakers}-{args.repetitions}-periods',
                     waveform=wave, mode=mode, speakers=args.speakers, fs=args.fs,
                     response_seconds=args.response_seconds, repetitions=args.repetitions,
                     gain=args.gain, f1=args.f1)


if __name__ == '__main__':
    main()
