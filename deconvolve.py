"""Deconvolve the exported playback WAV, retaining each recorded repetition.

Supports fresh-suite schedules (including unpadded October files) and cyclic
metadata from mimo_rir.py. Legacy ess/perfect/noise use mimo_rir.py deconv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import irfft, rfft


def schedule(meta):
    if meta.get('method') == 'cyclic':
        n = meta['speakers'] * meta['ir_len']
        meta = dict(sample_rate=meta['fs'], outputs=meta['speakers'],
                    response_samples=meta['ir_len'], repetitions=meta['periods'],
                    period_samples=n, mode='simultaneous',
                    measured_starts_samples=[[meta['pre_roll'] + p * n for p in range(meta['periods'])]])
    keys = ('sample_rate', 'outputs', 'response_samples', 'repetitions', 'period_samples')
    if any(not isinstance(meta.get(k), int) or isinstance(meta[k], bool) or meta[k] <= 0 for k in keys):
        raise ValueError('invalid schedule: positive integer sample counts, outputs and repetitions required')
    mode, n, length = meta.get('mode'), meta['period_samples'], meta['response_samples']
    if mode not in ('single-speaker', 'simultaneous'):
        raise ValueError('requires fresh-suite or cyclic metadata; use mimo_rir.py for legacy methods')
    if (mode == 'simultaneous' and n != meta['outputs'] * length) or (mode == 'single-speaker' and n < 2 * length):
        raise ValueError('period length does not match the response support and schedule')
    starts = meta.get('measured_starts_samples')
    blocks = meta['outputs'] if mode == 'single-speaker' else 1
    if (not isinstance(starts, list) or len(starts) != blocks or
            any(not isinstance(row, list) or len(row) != meta['repetitions'] for row in starts)):
        raise ValueError('invalid measured-period starts')
    flat = [v for row in starts for v in row]
    if (any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in flat) or
            any(b < a + n for a, b in zip(flat, flat[1:]))):
        raise ValueError('measured periods must be nonnegative, ordered and nonoverlapping')
    return meta


def deconvolve(metadata, recording, *, offset_samples=0, output=None, playback=None):
    metadata, recording = Path(metadata), Path(recording)
    if not isinstance(offset_samples, int) or offset_samples < 0:
        raise ValueError('offset must be a nonnegative integer')
    meta = schedule(json.loads(metadata.read_text()))
    playback = Path(playback) if playback is not None else metadata.with_suffix('.wav')
    output = Path(output) if output is not None else recording.with_name(recording.stem + '-rir-periods.npy')
    if output.suffix != '.npy':
        raise ValueError('output filename must end in .npy')
    sidecar = output.with_suffix('.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError('Output array or provenance JSON already exists')
    n, length, outputs = meta['period_samples'], meta['response_samples'], meta['outputs']
    starts = meta['measured_starts_samples']
    with sf.SoundFile(str(playback)) as source, sf.SoundFile(str(recording)) as rec:
        if source.samplerate != meta['sample_rate'] or rec.samplerate != source.samplerate:
            raise ValueError('playback and recording must use the metadata sample rate without resampling')
        if source.channels != outputs:
            raise ValueError('playback channel count does not match metadata')
        required = starts[-1][-1] + n
        if len(source) < required or len(rec) < required + offset_samples:
            raise ValueError(f'playback or recording is short: need {required} playback samples and {required + offset_samples} recording samples')
        source.seek(starts[0][0])
        reference = np.empty(n)
        for a in range(0, n, meta['sample_rate']):
            b = min(n, a + meta['sample_rate'])
            reference[a:b] = source.read(b - a, always_2d=True)[:, 0]
        spectrum = rfft(reference)
        if not np.all(np.isfinite(spectrum)) or np.any(abs(spectrum) == 0):
            raise ValueError('playback reference has a zero or nonfinite spectral bin')
        shape = (meta['repetitions'], rec.channels, outputs, length)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the path exclusively before creating the NPY header and mapping.
        with output.open('xb'):
            pass
        h = None
        try:
            h = np.lib.format.open_memmap(output, mode='w+', dtype='float32', shape=shape)
            for block, row in enumerate(starts):
                for p, start in enumerate(row):
                    rec.seek(offset_samples + start)
                    y = rec.read(n, always_2d=True)
                    if not np.all(np.isfinite(y)):
                        raise ValueError('recording contains nonfinite samples')
                    d = irfft(rfft(y, axis=0) / spectrum[:, None], n, axis=0)
                    if meta['mode'] == 'simultaneous':
                        h[p] = d.reshape(outputs, length, rec.channels).transpose(2, 0, 1)
                    else:
                        h[p, :, block] = d[:length].T
                h.flush()
            del h
            h = None
            with sidecar.open('x') as dest:
                json.dump(dict(metadata=str(metadata.resolve()), playback=str(playback.resolve()),
                               recording=str(recording.resolve()), offset_samples=offset_samples,
                               sample_rate=meta['sample_rate'], shape=shape, dtype='float32',
                               axes=['repetition', 'microphone', 'loudspeaker', 'sample'],
                               averaging=False, alignment=False,
                               reference='first measured period, exported playback channel 1',
                               spectrum_condition=float(abs(spectrum).max() / abs(spectrum).min())), dest, indent=2)
                dest.write('\n')
        except BaseException:
            if h is not None:
                del h
            output.unlink(missing_ok=True)
            raise
    print(f'{output}: {shape} [repetition, microphone, loudspeaker, sample]; no averaging')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--offset-samples', type=int, default=0,
                        help='recording sample corresponding to playback WAV sample zero, including measured latency correction')
    parser.add_argument('--playback', type=Path, help='override playback WAV location (default: beside metadata)')
    parser.add_argument('-o', '--output', type=Path)
    args = parser.parse_args()
    deconvolve(args.metadata, args.recording, offset_samples=args.offset_samples,
               output=args.output, playback=args.playback)


if __name__ == '__main__':
    main()
