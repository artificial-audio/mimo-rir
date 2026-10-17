"""MIMO RIR measurement prototype: excitation generation and deconvolution.

M loudspeakers play one multichannel WAV (from Reaper), K microphones are recorded
sample-synchronously on the same interface. Three methods:

  ess      exponential sweeps, one loudspeaker at a time (non-simultaneous reference)
  perfect  one periodic perfect sweep, circularly shifted by m*L for loudspeaker m
  noise    independent Gaussian white noise per loudspeaker, cross-correlation

  python mimo_rir.py gen perfect -o perfect          -> perfect.wav (play) + perfect.json (keep)
  python mimo_rir.py deconv perfect.json rec.wav     -> rec_ir.npy, shape [mics, speakers, L]
  python mimo_rir.py test

deconv regenerates the excitation from the .json, so the big WAV is never read back.
"""
import argparse
import json

import numpy as np
import soundfile as sf
from scipy.fft import irfft, next_fast_len, rfft


def ess(n, fs, f1=20.0, fade=0.01):
    """Exponential sine sweep f1 -> fs/2 over n samples, raised-cosine fades."""
    T, R = n / fs, np.log(fs / 2 / f1)
    x = np.sin(2 * np.pi * f1 * T / R * (np.exp(np.arange(n) / fs * R / T) - 1))
    w = np.hanning(2 * int(fade * fs))
    x[: len(w) // 2] *= w[: len(w) // 2]
    x[-(len(w) // 2):] *= w[len(w) // 2:]
    return x


def perfect_sweep(N):
    """Periodic linear sweep with exactly flat spectrum: circular autocorrelation is a delta."""
    k = np.arange(N // 2 + 1)
    x = irfft(np.exp(-2j * np.pi * k**2 / N), N)
    return x / np.abs(x).max()


def n_samples(meta):
    """Excitation length; the WAV adds ir_len samples of silence after it."""
    M, L = meta['speakers'], meta['ir_len']
    if meta['method'] == 'ess':
        return M * (meta['sweep_len'] + L)
    return meta['periods'] * M * L if meta['method'] == 'perfect' else meta['dur']


def excite(meta, a, b):
    """Excitation samples [a, b) for all loudspeakers, shape [b-a, M]. Deterministic from meta."""
    M, L, fs = meta['speakers'], meta['ir_len'], meta['fs']
    t = np.arange(a, b)
    if meta['method'] == 'ess':
        s = ess(meta['sweep_len'], fs, meta['f1'])
        slot, i = t // (len(s) + L), t % (len(s) + L)
        on = (i < len(s)) & (slot < M)
        x = np.zeros((b - a, M))
        x[on.nonzero()[0], slot[on]] = s[i[on]]
    elif meta['method'] == 'perfect':
        x = perfect_sweep(M * L)[(t[:, None] - L * np.arange(M)) % (M * L)]
    else:  # noise: one RNG stream per 1 s block, so any range can be regenerated
        c0, c1 = a // fs, -(-b // fs)
        x = np.concatenate([np.random.default_rng([meta['seed'], c]).standard_normal((fs, M)) for c in range(c0, c1)])
        x = np.clip(x[a - c0 * fs: b - c0 * fs] / 4, -1, 1)
    return meta['gain'] * x


def gen(meta, out):
    n, fs = n_samples(meta), meta['fs']
    # RF64: 128 channels pass the 4 GB WAV limit within a few minutes
    with sf.SoundFile(out + '.wav', 'w', fs, meta['speakers'], 'PCM_24', format='RF64') as f:
        for a in range(0, n, fs):
            f.write(excite(meta, a, min(a + fs, n)))
        f.write(np.zeros((meta['ir_len'], meta['speakers'])))
    with open(out + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(f"{out}.wav: {(n + meta['ir_len']) / fs:.1f} s, {meta['speakers']} ch")


def deconv(meta, y):
    """y: recording [n, mics], sample 0 = excitation start. Returns h [mics, M, L]."""
    M, L, fs, g = meta['speakers'], meta['ir_len'], meta['fs'], meta['gain']
    y = np.pad(y, ((0, max(0, n_samples(meta) + L - len(y))), (0, 0)))
    if meta['method'] == 'ess':
        s = ess(meta['sweep_len'], fs, meta['f1'])
        S, nfft = len(s) + L, next_fast_len(2 * len(s) + L)
        X = rfft(s, nfft)
        # ponytail: fixed Tikhonov inverse; band-limit to [f1, fs/2] if the out-of-band noise bothers
        inv = X.conj() / (np.abs(X) ** 2 + 1e-8 * np.abs(X).max() ** 2)
        h = np.stack([irfft(rfft(y[m * S:(m + 1) * S], nfft, axis=0) * inv[:, None], nfft, axis=0)[:L] for m in range(M)])
        return h.transpose(2, 0, 1) / g
    if meta['method'] == 'perfect':
        N, P = M * L, meta['periods']
        Y = y[N:P * N].reshape(P - 1, N, -1).mean(0)  # drop first period (transient), average the rest
        d = irfft(rfft(Y, axis=0) / rfft(perfect_sweep(N))[:, None], N, axis=0)
        return d.reshape(M, L, -1).transpose(2, 0, 1) / g  # tails longer than L leak into speaker m+1
    # noise: block-wise cross-correlation, normalised by each channel's energy
    C, n = fs, meta['dur']
    nfft = next_fast_len(C + L)
    R, E = 0, 0
    for a in range(0, n, C):
        x = excite(meta, a, min(a + C, n))
        R = R + rfft(x, nfft, axis=0).conj()[:, :, None] * rfft(y[a:a + C + L], nfft, axis=0)[:, None, :]
        E = E + (x ** 2).sum(0)
    return (irfft(R, nfft, axis=0)[:L] / E[:, None]).transpose(2, 1, 0)


def test():
    """Simulate a random MIMO system and check that each method recovers it."""
    from scipy.signal import oaconvolve
    rng = np.random.default_rng(0)
    fs, M, K, L = 8000, 4, 2, 200
    h = rng.standard_normal((K, M, L)) * np.exp(-np.arange(L) / 40)
    for extra, tol in [(dict(method='ess', sweep_len=fs, f1=20.0), 0.05),
                       (dict(method='perfect', periods=3), 1e-6),
                       (dict(method='noise', dur=30 * fs, seed=1), 0.15)]:
        meta = dict(fs=fs, speakers=M, ir_len=L, gain=0.5, **extra)
        n = n_samples(meta)
        x = excite(meta, 0, n)
        y = np.stack([sum(oaconvolve(x[:, m], h[k, m])[:n + L] for m in range(M)) for k in range(K)], 1)
        err = np.linalg.norm(deconv(meta, y) - h) / np.linalg.norm(h)
        print(f"{meta['method']:8s} rel. error {err:.2e}")
        assert err < tol, meta['method']
    print('ok')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('method', choices=['ess', 'perfect', 'noise'])
    g.add_argument('-o', '--out', required=True)
    g.add_argument('--fs', type=int, default=48000)
    g.add_argument('--speakers', type=int, default=128)
    g.add_argument('--ir-len', type=float, default=0.25, help='s; slot per speaker, sets the perfect-sweep period')
    g.add_argument('--gain', type=float, default=0.5)
    g.add_argument('--sweep-len', type=float, default=2.0, help='s (ess)')
    g.add_argument('--f1', type=float, default=20.0, help='Hz (ess)')
    g.add_argument('--periods', type=int, default=4, help='perfect; first is discarded')
    g.add_argument('--dur', type=float, default=60.0, help='s (noise)')
    g.add_argument('--seed', type=int, default=0, help='noise')
    d = sub.add_parser('deconv')
    d.add_argument('meta')
    d.add_argument('rec', help='recorded WAV, mics only')
    d.add_argument('--latency', type=int, default=0, help='samples to drop from the recording start (measure via loopback)')
    d.add_argument('-o', '--out')
    sub.add_parser('test')
    a = p.parse_args()
    if a.cmd == 'test':
        test()
    elif a.cmd == 'gen':
        meta = dict(method=a.method, fs=a.fs, speakers=a.speakers, ir_len=2 * round(a.ir_len * a.fs / 2), gain=a.gain)
        meta |= {'ess': dict(sweep_len=round(a.sweep_len * a.fs), f1=a.f1),
                 'perfect': dict(periods=a.periods),
                 'noise': dict(dur=round(a.dur * a.fs), seed=a.seed)}[a.method]
        gen(meta, a.out)
    else:
        meta = json.load(open(a.meta))
        y, fs = sf.read(a.rec, always_2d=True)
        assert fs == meta['fs'], f'recording at {fs} Hz, excitation at {meta["fs"]} Hz'
        h = deconv(meta, y[a.latency:])
        out = a.out or a.rec.rsplit('.', 1)[0] + '_ir.npy'
        np.save(out, h)
        print(f'{out}: {h.shape} [mics, speakers, samples]')
