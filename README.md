# MIMO room impulse-response measurements

Generate multichannel sweep playback, check loudspeaker routing with musical notes, and recover one impulse response for every microphone–loudspeaker pair. The AudioLab defaults are 128 outputs, 48 kHz, three repetitions and 1 s response support. Microphone count is read from the recording.

## Install and check

Use Python 3.10 or newer. SoundFile needs libsndfile; its current wheels normally include it on macOS and Windows. On Linux, install your distribution's `libsndfile1` if necessary.

```sh
git clone https://github.com/artificial-audio/mimo-rir.git
cd mimo-rir
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
python mimo_rir.py test
```

On Windows, activate with `.venv\Scripts\activate`. Alternatively, run commands with `uv run --with-requirements requirements.txt python ...` instead of activating a virtual environment. There is no audio-device control: play and record the generated WAVs in REAPER or another DAW.

## 1. Check physical output routing

```sh
python generate-musical-sequence.py -o data/routing/musical-speaker-check-128.wav
```

This produces a 128-channel RF64 PCM24 WAV, a JSON schedule and a CSV output/note map. Each loudspeaker plays one 250 ms sine note, in groups of eight ascending white keys; each new group's starting key advances by one white key (C3…C4, D3…D4, etc.). Total duration is 32.2 s, including exactly 100 ms digital silence at each end. All samples are checked for channel isolation, endpoint silence, level and approximate pitch. Change `--speakers`, `--note-ms`, `--start-octave`, `--gain` or `--fs` when needed.

Play at a low hardware level first. Route playback channel 1 to physical output 1, continuing through output 128; listen and record the actual output-to-speaker mapping, missing notes and unexpected locations. The generator verifies the file, while listening verifies the physical system. It does not automatically identify failed loudspeakers from a microphone recording. Quiet results can also come from routing, gains or a microphone fault.

## 2. Generate measurement playback

```sh
python generate-measurements.py --output-dir data/baseline
```

This generates `exponential-simultaneous-128-3-periods.wav`, its matching `.json` schedule, and `-verification.json`. Every exported sample is read back and checked against the design, including quantization, period identity and a finite-support inverse simulation. Keep the verification report with the WAV and schedule. Existing outputs are refused. A failed/incomplete generation should be inspected and regenerated under a new name; do not use a WAV without a passing verification report.

The default file is **386.22 s**, approximately **7.12 GB** in decimal bytes. All 128 outputs play simultaneously, with successive channels circularly shifted by 48,000 samples (1 s). One measured period is 128 s. Playback sample zero includes 100 ms leading silence; measured periods start at **1.11, 129.11 and 257.11 s**, samples **53,280; 6,197,280; 12,341,280**. There is a 10 ms fade plus 1 s periodic warm-up before measurement and a fade/support suffix, then 100 ms trailing silence. There are no gaps, fades or resets inside or between measured periods.

For all four comparison conditions:

```sh
python generate-measurements.py --output-dir data/comparison --waveform both --mode both
```

| Mode | Active outputs | Period | Repetitions | Default file duration |
|---|---|---|---|---|
| `simultaneous` | All outputs, shifted by response support | 128 s | Three total | 386.22 s |
| `single-speaker` | One output per block, in output order | 2 s | Three per speaker | 1026.76 s |

Single-speaker files are approximately 18.93 GB each; all four files need about 52.1 GB before recordings and results. Verification reads them completely. Small rehearsal files can use `--speakers 4 --fs 8000 --response-seconds 0.025`. Other options are `--waveform linear`, `--repetitions`, `--gain` and `--f1`; run `--help` for defaults. Response support is rounded to an even number of samples. The JSON's `measured_starts_samples` is authoritative for every configuration.

These are periodic sweeps for linear system identification. The linear variant has flat spectral magnitude; the exponential variant has exponential group delay above `f1`, pink spectral power and a nonzero low-frequency shelf. It is not an isolated Farina ESS measurement and does not separate nonlinear harmonic orders. Equal peak gain does not mean equal energy. Single-speaker and simultaneous modes have different period lengths and frequency trajectories, so their comparison does not isolate simultaneous playback alone. See [design and legacy methods](docs/design-and-legacy.md) for details.

## 3. Record in the lab

1. Set the interface, DAW and files to the same sample rate, normally 48 kHz. Verify clock synchronization across every digital converter. For the AudioLab chain, retain A32 WORD CLOCK OUT → OctaMic II WORD CLOCK IN and verify that the OctaMic follows the shared clock; this connection alone is not proof of a fault-free recording.
2. Import the complete multichannel WAV on a track with the required channel count. Preserve one-to-one output routing. Disable normalization, effects, time stretching, resampling, added fades and stereo downmixing. Play the whole file once, without looping it.
3. Record all microphones together continuously, from before playback starts until its end. Keep gains fixed and leave headroom. Log channel order, microphone positions, output mapping, hardware gains, clock settings, temperature and file names. Keep the original WAVs and DAW project.
4. Establish the recording sample corresponding to playback WAV sample zero. Use the DAW item positions and a measured loopback correction where available. Do not assume reported driver compensation is exact. The offset is specific to each take; do not reuse an old take's value.
5. Inspect the raw recording for clipping, bursts, interruptions and repeated/zero-valued sample blocks before averaging. Recent AudioLab recordings contained all-channel zero blocks as short as 16–32 samples, below 1 ms. Check at sample resolution and across channels. A disturbance common to all microphones is not evidence of a single faulty mic cable. Its hardware/software cause still needs diagnosis.

## 4. Deconvolve each repetition

```sh
python deconvolve.py data/baseline/exponential-simultaneous-128-3-periods.json data/recordings/baseline.wav --offset-samples 96000 -o results/baseline-rir-periods.npy
```

**96,000 is an example for a recording that starts 2 s before playback sample zero. Measure your own offset.** The extraction index is `offset_samples + measured_starts_samples[block][repetition]`. It already includes the leading silence and warm-up stored in the schedule. Do not add them again, and do not confuse DAW placement with acoustic propagation delay. If playback starts before the recording, re-export the recording with the appropriate leading timeline silence; negative offsets and truncated measured windows are rejected.

The matching exported WAV is read beside its JSON, or can be supplied with `--playback /path/to/original.wav`. Keep the original quantized playback WAV; regeneration is not a substitute for knowing what was played. Do not edit its channel order or schedule. The decoder reads the first measured period on playback channel 1 as its reference. Generated schedules guarantee matching periods and shifted channels; arbitrary playback files do not.

The output is float32 NPY with axes **[repetition, microphone, loudspeaker, sample]**, with zero-based array indices. Defaults with eight microphones give `[3, 8, 128, 48000]`, approximately **590 MB**. This is `3 × 8 × 128 × 48000 × 4` bytes plus the header. Ten microphones give approximately 737 MB. Only one recording period is processed at a time; full-size FFTs still require additional working memory, so allow several GB of RAM. Playback gain is already included in the actual reference; no extra gain division is applied. A neighboring JSON records input paths, sample rate, offset, axes and spectral condition. Outputs are never normalized, independently aligned or averaged, and existing results are refused.

Example access in Python:

```python
import numpy as np
h = np.load('results/baseline-rir-periods.npy', mmap_mode='r')
rir = h[0, 0, 39]  # repetition 1, microphone 1, loudspeaker 40
time_seconds = np.arange(rir.size) / 48000
```

## 5. Evaluate before combining repetitions

Keep the per-repetition array as the primary result. Compare channel activity and each path's early response, tail and repeatability at unchanged gain and delay. Energy or an early/late power contrast can flag weak paths, but is not a calibrated noise SNR or proof of a failed loudspeaker. Confirm suspicious outputs with the audible routing check and physical inspection. Retain explicit period/path masks and document every exclusion.

A digital interruption in a simultaneous period can contaminate responses for **all loudspeakers**, even when it is brief. Exclude affected periods from a stationary baseline rather than averaging them away. The 1 s response support must contain propagation, uncompensated latency and significant decay; energy beyond it leaks into adjacent output slots. Choose support from measured decay/noise-floor behavior with margin. Software simulation does not establish adequate support, clock stability or room stationarity.

For a moving microphone, the whole-period output is a **pseudo impulse response**. Frequencies represent different acquisition times; motion/Doppler and reflections can bend or broaden its spectrogram. Its apparent delay is not directly a position estimate. Moving-microphone tracking and the date-specific research analyses are outside this basic acquisition toolset.

## Compatibility and contents

| File | Role |
|---|---|
| `generate-measurements.py` | Preferred periodic generator, both waveforms and schedules, full read-back verification |
| `deconvolve.py` | Actual-WAV inversion, every repetition retained; fresh-suite and `cyclic` metadata |
| `generate-musical-sequence.py` | Audible output check and note/channel maps |
| `mimo_rir.py` | Shared periodic waveform design and original `ess`, `perfect`, `noise`, `cyclic` prototype |
| `test_measurements.py`, `test_cyclic.py` | Small quantized-WAV recovery, timing, channel and rejection checks |
| `.github/workflows/test.yml` | Runs the tests and legacy simulations on Python 3.10 and 3.12 |

The fresh-suite scripts were consolidated from the working AudioLab measurement folder on 2026-11-06. The unpublished October cyclic/exponential changes and their tests are retained. New generation adds the requested 100 ms endpoint silence and exposes the lab constants as command-line options. Historical October/November fresh-suite JSON files, which have **no endpoint padding**, remain readable: their stored sample indices are used unchanged. Do not replace an old schedule with a newly generated one.

Legacy `mimo_rir.py deconv` regenerates its reference and averages cyclic/perfect repetitions. Use `deconvolve.py` for new work and historical fresh-suite recordings. Legacy `ess`, `perfect` and `noise` metadata still use the original decoder; they are not interchangeable with the newer schedule. Legacy generators retain their historical timing for reproduction, including the absence of 100 ms endpoint silence; use the preferred generator above for new playback files.

Keep generated playback/recording WAVs, RIRs and result packages outside Git. `data/`, `results/` and audio/array files are ignored, but JSON is no longer globally ignored so reusable schedules can be versioned deliberately. Store each finished dataset with its matching WAV/JSON, original recordings, offset, channel map, quality masks, software commit and reports in the project's data storage. This repository contains no laboratory recordings or measurement-result binaries.
