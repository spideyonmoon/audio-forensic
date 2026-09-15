"""Accuracy regressions: python -X utf8 -m unittest -v test_accuracy.

FFmpeg integration checks are skipped when FFmpeg is unavailable. Fixtures are
deterministic controls, not an estimate of accuracy on a real music corpus.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_forensic import SpectralEngine


class AccuracyRegressionTests(unittest.TestCase):
    def setUp(self):
        self.sr = 44100
        self.engine = SpectralEngine(Path("unused.flac"), self.sr)
        self.noise = (np.random.default_rng(81).standard_normal(self.sr * 3)
                      * 0.2).astype(np.float32)

    def test_dual_mono_and_quiet_side_do_not_invent_aac_evidence(self):
        reference = self.engine._mdct_quant_error(self.noise, None)
        self.assertLess(reference, 0.06)
        for side in (np.zeros_like(self.noise), self.noise * 1e-7):
            with self.subTest(side_peak=float(np.max(np.abs(side)))):
                self.assertAlmostEqual(
                    self.engine._mdct_quant_error(self.noise, side), reference)

    def test_side_channel_uses_its_own_active_anchors(self):
        reference = self.engine._mdct_quant_error(self.noise, None)
        actual = self.engine._mdct_quant_error(np.zeros_like(self.noise), self.noise)
        self.assertAlmostEqual(actual, reference)

    def test_tones_do_not_invent_aac_evidence(self):
        t = np.arange(len(self.noise)) / self.sr
        for frequencies in ((440,), (440, 880, 1320)):
            with self.subTest(frequencies=frequencies):
                tone = sum(0.1 * np.sin(2 * np.pi * f * t) for f in frequencies)
                score = self.engine._mdct_quant_error(tone.astype(np.float32), None)
                self.assertLess(score, 0.06)

    def test_silence_and_very_quiet_audio_abstain(self):
        for audio in (np.zeros_like(self.noise), self.noise * 1e-7):
            self.assertEqual(self.engine._mdct_quant_error(audio, None), -1.0)

    def test_isolated_transient_is_insufficient(self):
        impulse = np.zeros_like(self.noise)
        impulse[self.sr] = 0.8
        self.assertEqual(self.engine._mdct_quant_error(impulse, None), -1.0)

    def test_silent_file_is_inconclusive(self):
        silence = np.zeros(self.sr, dtype=np.float32)
        with patch.object(self.engine, "_decode_stereo", return_value=(silence, silence)):
            result = self.engine.analyse()
        self.assertEqual(result.verdict_label, "INCONCLUSIVE")
        self.assertEqual(result.mdct_quant_score, -1.0)

    def test_dual_mono_full_engine_has_no_lattice_penalty(self):
        with patch.object(self.engine, "_decode_stereo",
                          return_value=(self.noise, np.zeros_like(self.noise))):
            result = self.engine.analyse()
        self.assertLess(result.mdct_quant_score, 0.06)
        self.assertLess(result.main_score, 55)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_real_aac_encodes_retain_lattice_evidence(self):
        # TNS is disabled deliberately: this detector does not model TNS.
        # Validate both supported rates and full-band high-bitrate encodes.
        rng = np.random.default_rng(34)
        with tempfile.TemporaryDirectory() as directory:
            for sr in (44100, 48000):
                n = sr * 6
                freq = np.fft.rfftfreq(n, 1 / sr)
                source = np.fft.irfft(
                    np.fft.rfft(rng.standard_normal(n)) / np.sqrt(np.maximum(freq, 40)), n=n)
                source = (source * (0.15 / np.std(source))).astype(np.float32)
                path = Path(directory) / "encoded.m4a"
                engine = SpectralEngine(path, sr, channels=1)
                self.assertLess(engine._mdct_quant_error(source, None), 0.06)
                for bitrate in ("256k", "320k"):
                    with self.subTest(sample_rate=sr, bitrate=bitrate):
                        subprocess.run([
                            "ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr),
                            "-ac", "1", "-i", "pipe:0", "-c:a", "aac", "-aac_tns", "0",
                            "-b:a", bitrate, str(path),
                        ], input=source.tobytes(), capture_output=True, check=True, timeout=30)
                        decoded = engine._decode_audio()
                        self.assertIsNotNone(decoded)
                        self.assertGreaterEqual(engine._mdct_quant_error(decoded, None), 0.10)


if __name__ == "__main__":
    unittest.main()
