# Design notes and legacy prototype

For new recordings, follow the [main workflow](../README.md). The commands below document the original prototype and its historical timing; they are retained for reproduction. Its generators lack the new 100 ms endpoint silence, and its decoder averages cyclic/perfect repetitions before quality checks. The preferred `deconvolve.py` uses the actual exported WAV and retains each repetition. Fresh-suite metadata is a separate schema and cannot be passed to the legacy decoder.

Prototype for simultaneous MIMO room impulse response measurement in the acoustic lab (128 loudspeakers, 8 microphones). One file: `mimo_rir.py`.

| method | excitation | deconvolution |
|---|---|---|
| `ess` | exponential sweeps, one loudspeaker after another (reference) | regularised spectral division per slot |
| `perfect` | one periodic perfect sweep (period `M·L`), loudspeaker *m* shifted by `m·L` | average steady-state periods, circular division, cut into `L`-sample slots |
| `noise` | independent Gaussian white noise per loudspeaker | block-wise cross-correlation |

```
python mimo_rir.py test                          # simulated 4x2 system, all three methods
python mimo_rir.py gen perfect -o perfect        # perfect.wav (128 ch, RF64) + perfect.json
python mimo_rir.py deconv perfect.json rec.wav   # rec_ir.npy, [mics, speakers, samples]
```

Defaults: 48 kHz, `--ir-len 0.25` s. Durations at 128 speakers: ess 288 s (5.3 GB), perfect 4 periods 128 s (2.4 GB), noise 60 s (1.1 GB). Keep the `.json`: `deconv` regenerates the excitation from it.

`--ir-len` is the hard limit for `perfect`: any tail longer than `L` leaks into the next loudspeaker's slot. Pick it from the lab's T60.

## One-period overlapped cyclic sweeps

```sh
uv run --with numpy --with scipy --with soundfile python mimo_rir.py gen cyclic -o cyclic-128 --ir-len 0.75
uv run --with numpy --with scipy --with soundfile python mimo_rir.py deconv cyclic-128.json recording.wav
```

Python entry point: `gen_cyclic_sweeps(out, speakers=128, ir_len=0.75, fs=48000, gain=0.5, periods=1)`. This writes an RF64 WAV and JSON and refuses to overwrite either file. Existing `gen perfect` and its metadata keep their original schedule and defaults.

All speakers play simultaneously, with channel `m` circularly shifted by `m * ir_len`. The waveform reuses the flat-spectrum cyclic sweep; it is not a new waveform claimed to fix the previous recording discrepancies. With 128 speakers, 48 kHz, and 0.75 s spacing, the measured period is exactly 96 s. By default there is **one measured period**, not four. The complete WAV is 97.52 s: 0.01 s fade-in, 0.75 s warm-up, 96 s measurement, 0.01 s fade-out, and 0.75 s silence. The warm-up is the periodic continuation preceding the measurement, not silence. Fades stay outside the measured period. To average more complete periods, explicitly pass `--periods 2` (or more); none are discarded.

Play the generated file **once**, record from its beginning through the end, and retain all microphone channels. Do not loop the entire file in REAPER: it contains warm-up, endpoint fades, and trailing silence. The measured periods themselves are exactly cyclic, with no inserted fade, phase restart, or silence at their internal wraps. Adjacent samples need not be equal at a wrap, particularly near Nyquist; the correct check is continuation of the same periodic sequence. Deconvolution uses the recorded prefix length in the JSON and rejects recordings missing any measured samples.

The spacing is the assumed finite **RIR support including propagation and any uncompensated latency**, not automatically T60. The response remaining beyond that support leaks into neighbouring output slots; choose support from a decay/noise-floor criterion with margin. The existing 0.75 s is a provisional lab setting, not a newly measured guarantee.

For a stationary, unrestricted causal FIR model with `M` inputs and `L` taps per input, each microphone has `M*L` unknowns and needs at least `M*L` independent scalar observations. A measured period of `M*L` samples with this flat-spectrum cyclic design gives an orthogonal square identification problem in the ideal synchronous case. No additional full-period warm-up is necessary if one response-length of periodic prehistory is provided. For the exact finite-support model, the retained window is already in periodic steady state after that prefix.

This is an observation-count bound, not a universal SNR optimum. Under additive white noise and equal fixed energy per input, orthogonal columns minimize least-squares noise amplification. Peak-limited signals can have different RMS/crest factors; coloured noise and frequency weighting change the energy allocation problem. Simultaneous playback also uses more total power if each loudspeaker retains its individual level. At a fixed total power budget, that energy increase is unavailable. RIR priors or restricted bandwidth change the effective parameter count.

Validation: `uv run --with numpy --with scipy --with soundfile python -m unittest test_cyclic` exercises real quantized WAV generation, one and three measured periods, circular shifts and wrap continuation, finite-start linear convolution with taps at the full support boundary, and short-recording rejection. `python mimo_rir.py test` retains the three legacy simulation checks. These tests do not validate playback hardware, room stationarity, or sample-clock synchronization.

## Exponential cyclic variant

`gen cyclic --waveform exponential --f1 20 --ir-len 0.75 --periods 3 -o cyclic-exp-128-three-periods` creates three consecutive 96 s measured periods in one 289.52 s file, with a single 0.76 s pre-roll and a 0.76 s fade-out/silence suffix. All 128 speakers play simultaneously; channel shifts are exactly 36,000 samples. Period starts are at samples 36,480, 4,644,480 and 9,252,480 (0.76, 96.76, 192.76 s). No interior fades or gaps are inserted. Play the complete file once and record continuously through its end.

This is a frequency-domain periodic exponential sweep: group delay follows `t(f) = T log(f/f1)/log(Nyquist/f1)` above `f1`, with pink spectral power (`1/f`). Below `f1`, a flat nonzero spectral magnitude shelf keeps the cyclic inverse finite; logarithmic phase continues into this shelf and wraps periodically. DC and Nyquist bins are real. Consequently the waveform differs from a finite, time-domain `ess()` sine with endpoint fades. Deconvolution uses this exact base waveform and the same slot separation; keep its JSON and use the updated code. Older code that assumes a linear perfect sweep cannot decode it correctly.

Nonflat spectra are invertible but the shifted columns are no longer orthogonal. At 48 kHz and `f1=20 Hz`, the ideal circulant condition number is `sqrt(24000/20) = 34.64`; high-frequency noise receives more inverse amplification than low-frequency noise. This does not establish an SNR improvement, nonlinear-distortion separation, sufficient 0.75 s physical support, or a cure for changing playback/record timing. Small quantized-WAV tests cover one/three periods, exact channel shifts and period identity, wrap continuation, spectral slope, and full-support FIR recovery.

## Reaper

1. Import the 128-channel WAV at position 0 on a track set to 128 channels; route track channels 1–128 to hardware outputs 1–128.
2. Record-arm an 8-channel track on mic inputs 1–8.
3. Record from 0 for the file length, then render/glue the recorded item from 0 as an 8-channel WAV at 48 kHz.
4. Reaper compensates the reported driver latency, but not exactly. Patch one output to one input as a loopback once, read the residual offset, and pass it as `deconv --latency <samples>`.

## Phase 2

Robust deconvolution (IRLS from `kyungyunlee/robust-deconv`, or the PyTorch variant) on the same recordings, plus MIMO least squares for `noise` instead of plain cross-correlation.
