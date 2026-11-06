"""Finite-record, quantized-WAV regression for the cyclic measurement schedule."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
from scipy.signal import oaconvolve

import mimo_rir as mimo


class CyclicTest(unittest.TestCase):
    def test_exponential_quantized_round_trip(self):
        rng = np.random.default_rng(19)
        with tempfile.TemporaryDirectory() as tmp:
            for periods in (1, 3):
                out = str(Path(tmp) / f'exponential-{periods}')
                meta = mimo.gen_cyclic_sweeps(out, speakers=4, ir_len=.025, fs=8000,
                                              periods=periods, waveform='exponential')
                self.assertEqual(json.loads(Path(out + '.json').read_text()), meta)
                x, fs = sf.read(out + '.wav', always_2d=True)
                L, N, start = meta['ir_len'], 4 * meta['ir_len'], meta['pre_roll']
                one = x[start:start+N]
                for m in range(4):
                    np.testing.assert_array_equal(one[:, m], np.roll(one[:, 0], m*L))
                for j in range(1, periods):
                    np.testing.assert_array_equal(one, x[start+j*N:start+(j+1)*N])
                np.testing.assert_array_equal(x[start-8:start], one[-8:])
                np.testing.assert_array_equal(x[start+periods*N], one[0])
                mag = abs(np.fft.rfft(one[:, 0]))
                freq = np.fft.rfftfreq(N, 1/fs)
                # Pink power gives 6.02 dB less spectral magnitude at 4x frequency.
                self.assertAlmostEqual(mag[np.argmin(abs(freq-100))] /
                                       mag[np.argmin(abs(freq-400))], 2., places=4)
                self.assertLess(mag.max()/mag.min(), np.sqrt((fs/2)/20)*1.001)
                h = rng.standard_normal((2,4,L))*np.exp(-np.arange(L)/40)
                h[:,:,-1] = .5
                y = np.stack([sum(oaconvolve(x[:,m],h[k,m]) for m in range(4))
                              for k in range(2)],axis=1)
                err=np.linalg.norm(mimo.deconv(meta,y)-h)/np.linalg.norm(h)
                self.assertLess(err,1e-5)
                print(f'exponential cyclic {periods} period(s): relative error {err:.2e}')

    def test_finite_record_round_trip(self):
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as tmp:
            for periods in (1, 3):
                out = str(Path(tmp) / f'cyclic-{periods}')
                meta = mimo.gen_cyclic_sweeps(out, speakers=4, ir_len=0.025,
                                               fs=8000, periods=periods)
                self.assertEqual(json.loads(Path(out + '.json').read_text()), meta)
                x, fs = sf.read(out + '.wav', always_2d=True)
                L, N = meta['ir_len'], 4 * meta['ir_len']
                self.assertEqual(fs, 8000)
                self.assertEqual(x.shape, (mimo.n_samples(meta) + L, 4))
                self.assertTrue(np.all(x[0] == 0))
                self.assertTrue(np.all(x[mimo.n_samples(meta) - 1:] == 0))
                self.assertLessEqual(np.max(np.abs(x)), meta['gain'] + 1.2e-7)
                start = meta['pre_roll']
                one = x[start:start + N]
                spectrum = np.abs(np.fft.rfft(one[:, 0]))
                self.assertLess(np.ptp(spectrum) / spectrum.mean(), 1e-4)
                for m in range(4):
                    np.testing.assert_array_equal(one[:, m], np.roll(one[:, 0], m * L))
                if periods > 1:
                    np.testing.assert_array_equal(x[start:start + N], x[start + N:start + 2 * N])
                # The pre-roll and post-measurement fade continue the SAME cyclic phase.
                np.testing.assert_array_equal(x[start - 8:start], one[-8:])
                np.testing.assert_array_equal(x[start + periods * N], one[0])
                h = rng.standard_normal((2, 4, L)) * np.exp(-np.arange(L) / 40)
                h[:, :, -1] = 0.5  # test full support; do not assume an already settled start
                y = np.stack([sum(oaconvolve(x[:, m], h[k, m]) for m in range(4))
                              for k in range(2)], axis=1)
                recovered = mimo.deconv(meta, y)
                error = np.linalg.norm(recovered - h) / np.linalg.norm(h)
                self.assertLess(error, 1e-5)
                print(f'cyclic {periods} measured period(s): relative error {error:.2e}')
                with self.assertRaisesRegex(ValueError, 'short'):
                    mimo.deconv(meta, y[:start + periods * N - 1])
                with self.assertRaises(FileExistsError):
                    mimo.gen_cyclic_sweeps(out)

    def test_bad_arguments_do_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / 'invalid')
            for kwargs in [dict(speakers=0), dict(periods=0), dict(fs=0), dict(periods=1.5),
                           dict(gain=0), dict(gain=2), dict(ir_len=float('nan')),
                           dict(ir_len=1e-10), dict(waveform='invalid'),
                           dict(waveform='exponential',f1=0),
                           dict(waveform='exponential',f1=24000),
                           dict(waveform='exponential',f1=float('nan'))]:
                with self.assertRaises(ValueError):
                    mimo.gen_cyclic_sweeps(out, **kwargs)
            self.assertFalse(Path(out + '.wav').exists())


if __name__ == '__main__':
    unittest.main()
