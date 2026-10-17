# mimo-rir

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

## Reaper

1. Import the 128-channel WAV at position 0 on a track set to 128 channels; route track channels 1–128 to hardware outputs 1–128.
2. Record-arm an 8-channel track on mic inputs 1–8.
3. Record from 0 for the file length, then render/glue the recorded item from 0 as an 8-channel WAV at 48 kHz.
4. Reaper compensates the reported driver latency, but not exactly. Patch one output to one input as a loopback once, read the residual offset, and pass it as `deconv --latency <samples>`.

## Phase 2

Robust deconvolution (IRLS from `kyungyunlee/robust-deconv`, or the PyTorch variant) on the same recordings, plus MIMO least squares for `noise` instead of plain cross-correlation.
