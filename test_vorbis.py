"""Vorbis regression fixtures; run python -X utf8 -m unittest -v test_vorbis.

Encodes are generated locally with libvorbis and transcoded to 24-bit FLAC.
This tests controlled cases, not a representative real-music benchmark.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_forensic import SpectralEngine


class VorbisTests(unittest.TestCase):
    @staticmethod
    def source(sr):
        rng = np.random.default_rng(17)
        n = sr * 4
        f = np.fft.rfftfreq(n, 1 / sr)
        channels = []
        for _ in range(2):
            x = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) /
                             np.sqrt(np.maximum(f, 40)), n=n)
            channels.append(x * 0.15 / np.std(x))
        return np.column_stack(channels).astype(np.float32)

    def test_genuine_controls(self):
        sr = 44100
        x = self.source(sr)[:, 0]
        t = np.arange(len(x)) / sr
        f = np.fft.rfftfreq(len(x), 1 / sr)
        controls = {
            "pink": x,
            "tone": 0.2 * np.sin(2 * np.pi * 440 * t),
            "harmonic": sum(0.2 / k * np.sin(2 * np.pi * 220 * k * t) for k in range(1, 20)),
            "brickwall": np.fft.irfft(np.fft.rfft(x) * (f < 16000), n=len(x)),
            "gradual_rolloff": np.fft.irfft(np.fft.rfft(x) / (1 + (f / 12000) ** 12), n=len(x)),
            "clipped": np.clip(x * 8, -0.4, 0.4),
            "quantized_16bit": np.round(x * 32768) / 32768,
            "silence": np.zeros_like(x),
        }
        engine = SpectralEngine(Path("unused.flac"), sr)
        for name, audio in controls.items():
            with self.subTest(control=name):
                score, _, _, _ = engine._vorbis_grid(audio, None)
                self.assertLess(score, 0.03, (name, score))

    def test_unsupported_and_short_audio_abstain(self):
        for sr, length in ((96000, 40000), (44100, 4000)):
            engine = SpectralEngine(Path("unused.flac"), sr)
            self.assertEqual(engine._vorbis_grid(np.ones(length), None)[0], -1)

    def test_zero_score_does_not_claim_proven_ancestry(self):
        engine = SpectralEngine(Path("unused.flac"), 44100)
        _, sentence, caveats = engine._verdict(0, 0, 21339, False)
        self.assertIn("unverified", sentence)
        self.assertTrue(any("does not prove" in item for item in caveats))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_transients_shift_long_block_alignment(self):
        # Abrupt attacks make libvorbis switch window sizes. Unlike stationary
        # noise, this fixture fails a detector requiring one global long phase.
        sr = 44100
        rng = np.random.default_rng(374)
        source = rng.standard_normal((sr * 12, 2)) * 0.012
        for seconds in (0.15, 0.85, 1.7, 2.4, 3.15, 3.8, 4.65, 5.25,
                        6.2, 7.1, 7.8, 8.9, 9.7, 10.3, 11.1):
            start, length = int(seconds * sr), int(0.35 * sr)
            source[start:start + length] += (
                rng.standard_normal((length, 2)) * 0.17 *
                np.exp(-np.arange(length)[:, None] / (sr * 0.09)))
        source = source.astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transients.ogg"
            engine = SpectralEngine(path, sr)
            clean = engine._vorbis_grid((source[:, 0] + source[:, 1]) / 2,
                                        (source[:, 0] - source[:, 1]) / 2)
            self.assertLess(clean[0], 0.03, clean)
            for quality in (6, 8):
                with self.subTest(quality=quality):
                    subprocess.run([
                        "ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr),
                        "-ac", "2", "-i", "pipe:0", "-c:a", "libvorbis", "-q:a", str(quality), str(path),
                    ], input=source.tobytes(), capture_output=True, check=True, timeout=30)
                    mid, side = engine._decode_stereo()
                    result = engine._vorbis_grid(mid, side)
                    self.assertGreaterEqual(result[0], 0.03, result)
                    self.assertGreaterEqual(result[1], 6, result)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_vorbis_to_flac_quality_and_rate_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            ogg = Path(directory) / "encoded.ogg"
            flac = Path(directory) / "transcoded.flac"
            for sr in (44100, 48000):
                source = self.source(sr)
                engine = SpectralEngine(flac, sr)
                self.assertLess(engine._vorbis_grid(
                    (source[:, 0] + source[:, 1]) / 2,
                    (source[:, 0] - source[:, 1]) / 2)[0], 0.03)
                for quality in (4, 6, 8, 10):
                    with self.subTest(sample_rate=sr, quality=quality):
                        subprocess.run([
                            "ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr),
                            "-ac", "2", "-i", "pipe:0", "-c:a", "libvorbis", "-q:a", str(quality), str(ogg),
                        ], input=source.tobytes(), capture_output=True, check=True, timeout=30)
                        subprocess.run([
                            "ffmpeg", "-v", "error", "-y", "-i", str(ogg),
                            "-c:a", "flac", "-sample_fmt", "s32", str(flac),
                        ], capture_output=True, check=True, timeout=30)
                        mid, side = engine._decode_stereo()
                        score, support, tested, _ = engine._vorbis_grid(mid, side)
                        self.assertGreaterEqual(score, 0.03, (score, support, tested))
                        self.assertGreaterEqual(support, max(4, (tested + 1) // 2))
                        if quality == 10:
                            # Arbitrary sample trimming and gain must not depend
                            # on the encoder delay or the original amplitude.
                            shifted = engine._vorbis_grid(mid[137:] * 0.37, side[137:] * 0.37)
                            self.assertGreaterEqual(shifted[0], 0.03, shifted)
                            with patch.object(engine, "_decode_stereo", return_value=(mid, side)):
                                result = engine.analyse()
                            self.assertGreaterEqual(result.main_score, 55)
                            self.assertTrue(any("Vorbis Transform Grid" in e for e in result.evidence))


if __name__ == "__main__":
    unittest.main()
