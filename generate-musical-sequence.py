"""One loudspeaker per note: eight ascending white keys, next group starts one key higher.

Requires numpy and soundfile. Writes a multichannel RF64 PCM24 WAV, a JSON
schedule and a CSV channel/note list. Adds 100 ms digital silence at each end.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

WHITE_KEYS = (('C', 0), ('D', 2), ('E', 4), ('F', 5), ('G', 7), ('A', 9), ('B', 11))


def note_for_channel(channel_zero_based, start_octave):
    # Each group spans an octave; its starting white key advances one step.
    white_index = channel_zero_based // 8 + channel_zero_based % 8
    octaves, step = divmod(white_index, 7)
    name, semitone = WHITE_KEYS[step]
    octave = start_octave + octaves
    midi = 12*(octave+1) + semitone
    return f'{name}{octave}', 440*2**((midi-69)/12)


def generate(out, *, speakers=128, fs=48000, note_ms=250., start_octave=3, gain=.5):
    out = Path(out)
    if out.suffix.lower() != '.wav':
        raise ValueError('Output filename must end with .wav')
    targets = [out, out.with_suffix('.json'), out.with_suffix('.csv')]
    if any(p.exists() for p in targets):
        raise FileExistsError('Output WAV, JSON or CSV already exists; choose a new name')
    if (any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (speakers, fs))
            or not np.isfinite(note_ms) or note_ms < 20 or round(.01 * fs) < 2):
        raise ValueError('speakers and fs must be positive; notes must be at least 20 ms')
    if not np.isfinite(gain) or not 0 < gain <= 1:
        raise ValueError('gain must be in (0, 1]')
    notes = [note_for_channel(m, start_octave) for m in range(speakers)]
    if max(f for _, f in notes) >= fs/2:
        raise ValueError('Highest requested note reaches Nyquist; reduce starting octave or speaker count')
    count = round(note_ms*fs/1000)
    silence = round(.1*fs)
    fade = min(round(.01*fs), count//2)
    ramp = .5-.5*np.cos(np.linspace(0, np.pi, fade))
    envelope = np.ones(count)
    envelope[:fade] = ramp
    envelope[-fade:] = ramp[::-1]
    time = np.arange(count)/fs
    schedule = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(out), 'x', fs, speakers, subtype='PCM_24', format='RF64') as wav:
        wav.write(np.zeros((silence, speakers)))
        for m, (name, frequency) in enumerate(notes):
            block = np.zeros((count, speakers))
            block[:, m] = gain*envelope*np.sin(2*np.pi*frequency*time)
            wav.write(block)
            schedule.append(dict(loudspeaker=m+1, group=m//8+1, note_in_group=m%8+1,
                note=name, frequency_hz=frequency, start_sample=silence+m*count,
                start_seconds=(silence+m*count)/fs, samples=count))
        wav.write(np.zeros((silence, speakers)))

    # Check the exported PCM file, including silence, channel isolation and pitch.
    with sf.SoundFile(str(out)) as wav:
        assert len(wav) == 2*silence+speakers*count and wav.channels == speakers
        assert wav.samplerate == fs and wav.subtype == 'PCM_24'
        assert not np.any(wav.read(silence, dtype='int32', always_2d=True))
        max_pitch_error = 0.
        peak = 0.
        for m, (_, frequency) in enumerate(notes):
            block = wav.read(count, always_2d=True)
            assert np.any(block[:, m])
            assert np.count_nonzero(block) == np.count_nonzero(block[:, m])
            peak = max(peak, float(np.max(abs(block))))
            nfft = 2**int(np.ceil(np.log2(count*8)))
            estimated = np.argmax(abs(np.fft.rfft(block[:, m], nfft)))*fs/nfft
            error = abs(estimated-frequency)
            assert error < fs/count, (m+1, estimated, frequency)
            max_pitch_error = max(max_pitch_error, error)
        assert not np.any(wav.read(silence, dtype='int32', always_2d=True))
        assert peak <= gain+1.2e-7
    digest = hashlib.sha256()
    with out.open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            digest.update(chunk)
    metadata = dict(sample_rate_hz=fs, channels=speakers, subtype='PCM_24', format='RF64',
        group_size=8, groups=(speakers+7)//8, start_octave=start_octave,
        note_samples=count, note_seconds=count/fs, leading_silence_samples=silence,
        trailing_silence_samples=silence, fade_samples=fade, peak_gain=gain,
        duration_seconds=(2*silence+speakers*count)/fs,
        timbre='Sine tones, 10 ms raised-cosine attack and release within each note slot',
        progression='Eight consecutive ascending white keys per group; next group starts one white key higher',
        schedule=schedule, verification=dict(all_channels_active_once=True,
        no_overlapping_active_channels=True, endpoint_silence_exact=True,
        peak_amplitude=peak, maximum_fft_pitch_error_hz=max_pitch_error,
        wav_sha256=digest.hexdigest()))
    out.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    with out.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(schedule[0]))
        writer.writeheader()
        writer.writerows(schedule)
    print(f'{out}: {speakers} channels, {metadata["duration_seconds"]:.3f} s; '
          f'{notes[0][0]} to {notes[-1][0]}; all checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speakers', type=int, default=128)
    parser.add_argument('--fs', type=int, default=48000)
    parser.add_argument('--note-ms', type=float, default=250.)
    parser.add_argument('--start-octave', type=int, default=3, help='Starting C octave; C3 is 130.81 Hz')
    parser.add_argument('--gain', type=float, default=.5)
    parser.add_argument('-o', '--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path(__file__).resolve().parent/f'musical-speaker-check-{args.speakers}.wav'
    generate(output, speakers=args.speakers, fs=args.fs, note_ms=args.note_ms,
             start_octave=args.start_octave, gain=args.gain)
