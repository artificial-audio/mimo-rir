"""Small exported-WAV/CLI checks; no laboratory hardware or large data required."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
from scipy.signal import oaconvolve

import deconvolve
import mimo_rir

ROOT = Path(__file__).resolve().parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generation = load_script('generate-measurements')
music = load_script('generate-musical-sequence')


class MeasurementTests(unittest.TestCase):
    def test_exported_round_trips_and_period_changes(self):
        rng = np.random.default_rng(23)
        fs, length, outputs, repetitions, offset = 8000, 200, 4, 3, 73
        truth = rng.normal(size=(2, outputs, length)) * np.exp(-np.arange(length) / 40)
        truth[:, :, -1] = .25
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            for waveform in ('linear', 'exponential'):
                for mode in ('single-speaker', 'simultaneous'):
                    with self.subTest(waveform=waveform, mode=mode):
                        stem = tmp / f'{waveform}-{mode}'
                        meta = generation.generate(stem, waveform=waveform, mode=mode,
                                                   speakers=outputs, fs=fs, response_seconds=length / fs,
                                                   repetitions=repetitions)
                        x, _ = sf.read(str(stem.with_suffix('.wav')), always_2d=True)
                        self.assertFalse(np.any(x[:800]))
                        self.assertFalse(np.any(x[-800:]))
                        first = meta['measured_starts_samples'][0][0]
                        n = meta['period_samples']
                        if mode == 'simultaneous':
                            for m in range(outputs):
                                np.testing.assert_array_equal(x[first:first+n, m], np.roll(x[first:first+n, 0], m * length))
                        # Full finite-start convolution includes a tap at the support boundary.
                        y = np.column_stack([sum(oaconvolve(x[:, m], truth[k, m]) for m in range(outputs)) for k in range(2)])
                        # A gain change in one period must survive, rather than being averaged away.
                        for row in meta['measured_starts_samples']:
                            y[row[1]:row[1] + n] *= .7
                        rec = tmp / f'{waveform}-{mode}-recording.wav'
                        sf.write(str(rec), np.pad(y, ((offset, 0), (0, 0))), fs, subtype='DOUBLE')
                        result = tmp / f'{waveform}-{mode}-h.npy'
                        subprocess.run([sys.executable, str(ROOT / 'deconvolve.py'), str(stem.with_suffix('.json')),
                                        str(rec), '--offset-samples', str(offset), '-o', str(result)], check=True)
                        h = np.load(result)
                        expected = np.stack([truth, .7 * truth, truth])
                        self.assertEqual(h.shape, expected.shape)
                        error = np.linalg.norm(h - expected) / np.linalg.norm(expected)
                        self.assertLess(error, 1e-6)
                        print(f'{waveform} {mode}: relative error {error:.2e}')
                        with self.assertRaises(FileExistsError):
                            deconvolve.deconvolve(stem.with_suffix('.json'), rec, output=result)
                        with self.assertRaises(FileExistsError):
                            generation.generate(stem)
                        with self.assertRaisesRegex(ValueError, 'short'):
                            deconvolve.deconvolve(stem.with_suffix('.json'), rec, offset_samples=len(y))

    def test_unpadded_historical_schedule(self):
        # October metadata has neither schema_version nor boundary-padding fields.
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            meta = generation.make_metadata(speakers=2, fs=8000, response_seconds=.025)
            meta['frames'] -= 2 * meta['leading_silence_samples']
            meta['measured_starts_samples'] = [[v - meta['leading_silence_samples'] for v in row]
                                              for row in meta['measured_starts_samples']]
            meta['leading_silence_samples'] = 0
            base = mimo_rir.periodic_sweep(meta['generator'])
            x = generation.samples(meta, base, 0, meta['frames'])
            for key in ('schema_version', 'leading_silence_samples', 'trailing_silence_samples'):
                meta.pop(key)
            (tmp / 'old.json').write_text(json.dumps(meta))
            sf.write(str(tmp / 'old.wav'), x, 8000, subtype='PCM_24')
            # A unit tap on source 1 should decode with no gain division or shift.
            sf.write(str(tmp / 'rec.wav'), x[:, :1], 8000, subtype='PCM_24')
            h = np.load(deconvolve.deconvolve(tmp / 'old.json', tmp / 'rec.wav'))
            expected = np.zeros_like(h)
            expected[:, 0, 0, 0] = 1
            np.testing.assert_allclose(h, expected, atol=1e-6)

    def test_cyclic_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            meta = mimo_rir.gen_cyclic_sweeps(str(tmp / 'cyclic'), speakers=2, fs=8000,
                                            ir_len=.025, waveform='exponential', periods=3)
            x, _ = sf.read(str(tmp / 'cyclic.wav'), always_2d=True)
            sf.write(str(tmp / 'rec.wav'), x[:, :1], 8000, subtype='PCM_24')
            h = np.load(deconvolve.deconvolve(tmp / 'cyclic.json', tmp / 'rec.wav'))
            self.assertEqual(h.shape, (3, 1, 2, meta['ir_len']))
            expected = np.zeros_like(h)
            expected[:, 0, 0, 0] = 1
            np.testing.assert_allclose(h, expected, atol=1e-6)

    def test_reject_bad_metadata_and_inputs(self):
        meta = generation.make_metadata(speakers=4, fs=8000, response_seconds=.025)
        for changes in ({'period_samples': 1}, {'outputs': 0}, {'repetitions': 1.5},
                        {'mode': 'ess'}, {'measured_starts_samples': [[-1, 1, 2]]},
                        {'measured_starts_samples': [[200, 201, 202]]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                deconvolve.schedule(dict(meta, **changes))
        for changes in ({'speakers': 0}, {'fs': 0}, {'repetitions': 1.5}, {'gain': 0},
                        {'response_seconds': float('nan')}, {'f1': 24000}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                generation.make_metadata(**changes)
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            generation.generate(tmp / 'sweep', speakers=4, fs=8000, response_seconds=.025)
            for rate, channels, data in ((16000, 1, np.ones(5000)),
                                         (8000, 1, np.full(5000, np.nan))):
                sf.write(str(tmp / 'rec.wav'), data, rate, subtype='DOUBLE')
                with self.assertRaises(ValueError):
                    deconvolve.deconvolve(tmp / 'sweep.json', tmp / 'rec.wav')
                self.assertFalse((tmp / 'rec-rir-periods.npy').exists())
            sf.write(str(tmp / 'rec.wav'), np.ones(5000), 8000, subtype='DOUBLE')
            sf.write(str(tmp / 'sweep.wav'), np.zeros((5000, 4)), 8000, subtype='PCM_24')
            with self.assertRaisesRegex(ValueError, 'spectral bin'):
                deconvolve.deconvolve(tmp / 'sweep.json', tmp / 'rec.wav')

    def test_musical_check(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'music.wav'
            music.generate(out, speakers=128, fs=8000, note_ms=25.)
            meta = json.loads(out.with_suffix('.json').read_text())
            self.assertEqual(len(meta['schedule']), 128)
            self.assertEqual(meta['schedule'][0]['note'], 'C3')
            self.assertEqual(meta['schedule'][7]['note'], 'C4')
            self.assertEqual(meta['schedule'][8]['note'], 'D3')
            self.assertTrue(meta['verification']['no_overlapping_active_channels'])
            with self.assertRaises(FileExistsError):
                music.generate(out)
            with self.assertRaises(ValueError):
                music.generate(out.with_name('bad.wav'), speakers=0)

    def test_lab_default_timing(self):
        meta = generation.make_metadata()
        self.assertEqual(meta['period_samples'], 6144000)
        self.assertEqual(meta['measured_starts_samples'], [[53280, 6197280, 12341280]])
        self.assertAlmostEqual(meta['frames'] / 48000, 386.22)


if __name__ == '__main__':
    unittest.main()
