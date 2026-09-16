"""Deterministic regression tests for the structural MQA detector."""
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from audio_forensic import (
    AudioTags,
    AudioTechnical,
    MQADetection,
    SpectralAnalysis,
    _MQA_MAGIC,
    _apply_mqa_override,
    _mqa_original_sample_rate,
    _scan_mqa_pcm,
    inspect_mqa,
)


def _bits(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def _fixture(bit_depth: int, *, rate_code: int = 9, provenance: int = 17,
             sync_start: int = 40, length: int = 256) -> tuple[np.ndarray, int]:
    """Build stereo integer PCM carrying one reverse-engineered control packet."""
    samples = np.zeros((length, 2), dtype=np.int32)
    plane = bit_depth - 16
    sync_end = sync_start + 35

    for offset, bit in enumerate(_bits(_MQA_MAGIC, 36)):
        samples[sync_start + offset, 0] |= bit << plane
    for offset, bit in enumerate(_bits(rate_code, 4)):
        samples[sync_end + 3 + offset, 0] |= bit << plane
    for offset, bit in enumerate(_bits(provenance, 5)):
        samples[sync_end + 29 + offset, 0] |= bit << plane
    return samples, sync_end


class MQADetectorTests(unittest.TestCase):
    def test_original_sample_rate_codes(self):
        expected = {
            0: 44100, 1: 48000, 8: 88200, 9: 96000,
            4: 176400, 5: 192000, 12: 352800, 13: 384000,
        }
        for code, rate in expected.items():
            with self.subTest(code=code):
                self.assertEqual(_mqa_original_sample_rate(code), rate)

    def test_detects_16_bit_studio_and_payload(self):
        samples, sync_end = _fixture(16, rate_code=9, provenance=17)
        result = _scan_mqa_pcm(samples, 16)
        self.assertTrue(result.detected)
        self.assertTrue(result.studio)
        self.assertEqual(result.original_sample_rate, 96000)
        self.assertEqual(result.bit_plane, 0)
        self.assertEqual(result.sync_sample, sync_end)

    def test_detects_24_bit_at_source_plane_eight(self):
        samples, sync_end = _fixture(24, rate_code=12, provenance=8)
        result = _scan_mqa_pcm(samples, 24, metadata_claimed=True)
        self.assertTrue(result.detected)
        self.assertTrue(result.metadata_claimed)
        self.assertFalse(result.studio)
        self.assertEqual(result.original_sample_rate, 352800)
        self.assertEqual(result.bit_plane, 8)
        self.assertEqual(result.sync_sample, sync_end)

    def test_searches_all_three_candidate_planes(self):
        samples, sync_end = _fixture(16)
        samples <<= 2
        result = _scan_mqa_pcm(samples, 16)
        self.assertTrue(result.detected)
        self.assertEqual(result.bit_plane, 2)
        self.assertEqual(result.sync_sample, sync_end)

    def test_metadata_claim_is_not_structural_detection(self):
        result = _scan_mqa_pcm(np.zeros((256, 2), dtype=np.int32), 16,
                               metadata_claimed=True)
        self.assertFalse(result.detected)
        self.assertTrue(result.metadata_claimed)

    def test_structural_hit_forces_known_lossy_100(self):
        spectral = SpectralAnalysis(
            main_score=0,
            heuristic_score=0,
            net_confidence_pct=0,
            verdict_label="GENUINE",
            primary_verdict="clean spectrum",
            natural_evidence=["natural-looking spectrum"],
        )
        _apply_mqa_override(spectral, MQADetection(
            detected=True,
            studio=True,
            original_sample_rate=96000,
            bit_plane=8,
            sync_sample=39,
        ))
        self.assertEqual(spectral.main_score, 100)
        self.assertEqual(spectral.net_confidence_pct, 100.0)
        self.assertEqual(spectral.verdict_label, "KNOWN_LOSSY")
        self.assertEqual(spectral.known_lossy_codec, "MQA Studio")
        self.assertIn("known lossy encoding", spectral.primary_verdict)
        self.assertTrue(any("FLAC preserves" in item for item in spectral.evidence))

    def test_metadata_only_does_not_force_lossy_override(self):
        spectral = SpectralAnalysis(main_score=15, heuristic_score=15,
                                    verdict_label="LIKELY_GENUINE")
        _apply_mqa_override(spectral, MQADetection(metadata_claimed=True))
        self.assertEqual(spectral.main_score, 15)
        self.assertEqual(spectral.verdict_label, "LIKELY_GENUINE")
        self.assertEqual(spectral.known_lossy_codec, "")

    def test_truncated_payload_still_reports_sync_without_guessing_fields(self):
        samples, sync_end = _fixture(16)
        result = _scan_mqa_pcm(samples[:sync_end + 1], 16)
        self.assertTrue(result.detected)
        self.assertIsNone(result.studio)
        self.assertEqual(result.original_sample_rate, 0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_ffmpeg_decode_preserves_the_signalling_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            for bit_depth, base in ((16, -8192), (24, -1048576)):
                with self.subTest(bit_depth=bit_depth):
                    samples, sync_end = _fixture(
                        bit_depth, rate_code=5, provenance=17, length=4096)
                    # Exercise signed words as well as the embedded control packet.
                    samples[:, 0] |= np.int32(base)
                    samples[:, 1] |= np.int32(base)
                    path = Path(directory) / f"synthetic-mqa-{bit_depth}.wav"
                    if bit_depth == 16:
                        pcm = samples.astype("<i2").tobytes()
                    else:
                        unsigned = samples.astype(np.int64) & 0xffffff
                        packed = np.empty((*samples.shape, 3), dtype=np.uint8)
                        packed[..., 0] = unsigned & 0xff
                        packed[..., 1] = (unsigned >> 8) & 0xff
                        packed[..., 2] = (unsigned >> 16) & 0xff
                        pcm = packed.tobytes()
                    with wave.open(str(path), "wb") as output:
                        output.setnchannels(2)
                        output.setsampwidth(bit_depth // 8)
                        output.setframerate(44100)
                        output.writeframes(pcm)
                    result = inspect_mqa(
                        AudioTags(),
                        AudioTechnical(precision=f"{bit_depth}-bit", channels="2"),
                        path,
                    )
                    self.assertTrue(result.detected)
                    self.assertTrue(result.studio)
                    self.assertEqual(result.original_sample_rate, 192000)
                    self.assertEqual(result.bit_plane, bit_depth - 16)
                    self.assertEqual(result.sync_sample, sync_end)


if __name__ == "__main__":
    unittest.main()
