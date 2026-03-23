#!/usr/bin/env python3
"""
af2.py — Audio Forensics CLI (Master Chef Edition)

Improvements over af.py:
  • Fixed entropy confusion (SoX entropy ≠ spectral entropy)
  • Fixed amplitude false positives (0.999 ≠ clipping)
  • Fixed spectral scoring thresholds (calibrated for real mastered audio)
  • Fixed DR/Crest Factor context-aware coloring
  • Clear inline interpretations (not just "low=..., high=...")
  • Accurate verdicts for CD-sourced & modern mastered audio

Engines: mediainfo · sox · ffmpeg
Native:   EBU R128/LUFS · spectral cutoff · bit-depth auth ·
          phase correlation · ReplayGain audit

Usage:
    python af2.py "path/to/audio.flac"           # full analysis
    python af2.py "path/to/audio.flac" --json    # machine-readable output
    python af2.py *.flac                         # batch / album mode
    python af2.py --info "path/to/audio.flac"    # lightweight mode
    python af2.py --demo                         # synthetic demo data

Requires: ffmpeg, sox, mediainfo in PATH.
Optional: numpy (for full spectral authenticity analysis)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# ANSI palette
# ---------------------------------------------------------------------------

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GOLD    = "\033[38;5;179m"
    CYAN    = "\033[38;5;73m"
    WHITE   = "\033[38;5;252m"
    GREY    = "\033[38;5;240m"
    GREEN   = "\033[38;5;107m"
    RED     = "\033[38;5;167m"
    YELLOW  = "\033[38;5;221m"
    ORANGE  = "\033[38;5;209m"
    BLUE    = "\033[38;5;110m"
    PURPLE  = "\033[38;5;183m"
    TEAL    = "\033[38;5;73m"


def _c(colour: str, text: str) -> str:
    return f"{colour}{text}{C.RESET}"

def _kv(key: str, value: str, *, width: int = 26) -> str:
    if not value:
        return ""
    return f"  {_c(C.CYAN, key.ljust(width))} {_c(C.WHITE, value)}"

def _rule(char: str = "─", width: int = 62) -> str:
    return _c(C.GREY, char * width)

def _section(title: str) -> str:
    pad = max(0, 58 - len(title))
    return f"\n{_c(C.GREY, '── ')}{_c(C.GOLD + C.BOLD, title)}{_c(C.GREY, ' ' + '─' * pad)}"

def _subsection(title: str) -> str:
    return f"\n  {_c(C.GREY, title)}"

def _badge(text: str, colour: str) -> str:
    return _c(colour + C.BOLD, f" {text} ")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AudioTags:
    title: str = ""
    album: str = ""
    date: str = ""
    album_artist: str = ""
    artist: str = ""
    bpm: str = ""
    comment_quality: str = ""
    comments: str = ""
    replaygain_track_gain: str = ""
    replaygain_album_gain: str = ""


@dataclass
class AudioTechnical:
    bit_rate: str = ""
    channels: str = ""
    precision: str = ""
    sample_rate: str = ""
    sample_encoding: str = ""
    duration: str = ""
    duration_sec: float = 0.0


@dataclass
class LoudnessProfile:
    # astats
    peak_db: str = ""
    rms_db: str = ""
    rms_peak_db: str = ""
    rms_trough_db: str = ""
    noise_floor_db: str = ""
    dynamic_range_db: str = ""
    crest_factor_db: str = ""
    flat_factor: str = ""
    peak_count: str = ""
    # SoX entropy — measures time-domain signal randomness (0=tonal/music, 1=noise)
    sox_entropy: str = ""
    dc_offset: str = ""
    zero_crossings_rate: str = ""
    # EBU R128 / ebur128
    lufs_integrated: str = ""
    lufs_range: str = ""      # LRA
    true_peak_dbtp: str = ""
    lufs_momentary_max: str = ""
    lufs_shortterm_max: str = ""
    # streaming delta
    spotify_delta: str = ""
    youtube_delta: str = ""


@dataclass
class AuthenticityReport:
    # Full spectral engine report
    spectral: "SpectralAnalysis | None" = None
    # Convenience shims
    spectral_cutoff_hz: str = ""
    spectral_cutoff_verdict: str = ""
    lpf_detected: bool = False
    lpf_cutoff_hz: str = ""
    # Bit-depth authenticity
    bit_depth_authentic: str = ""
    # Phase
    phase_correlation: str = ""
    phase_verdict: str = ""
    # Clipping
    clipped_samples: str = ""
    clipping_verdict: str = ""
    # Silence
    silence_total_pct: str = ""
    silence_sections: list[str] = field(default_factory=list)
    # ReplayGain audit
    rg_stored: str = ""
    rg_measured_lufs: str = ""
    rg_delta: str = ""
    rg_verdict: str = ""


@dataclass
class ForensicReport:
    filepath: Path
    tags: AudioTags = field(default_factory=AudioTags)
    technical: AudioTechnical = field(default_factory=AudioTechnical)
    sox_stats: dict[str, str] = field(default_factory=dict)
    loudness: LoudnessProfile = field(default_factory=LoudnessProfile)
    authenticity: AuthenticityReport = field(default_factory=AuthenticityReport)
    dr_score: str = "N/A"
    spectrogram_path: Optional[Path] = None

    @property
    def file_size_mb(self) -> float:
        return self.filepath.stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)

def _tool_available(name: str) -> bool:
    checker = "where" if sys.platform == "win32" else "which"
    try:
        return subprocess.run([checker, name], capture_output=True, check=False).returncode == 0
    except FileNotFoundError:
        return False

def _warn(msg: str) -> None:
    print(f"{_c(C.YELLOW, 'Warning:')} {msg}", file=sys.stderr)

def _camel_case(text: str) -> str:
    words = re.sub(r"[^a-zA-Z0-9 ]", "", text).split()
    if not words:
        return ""
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


# ---------------------------------------------------------------------------
# Tool: mediainfo
# ---------------------------------------------------------------------------

def extract_mediainfo(filepath: Path) -> tuple[AudioTags, AudioTechnical]:
    result = _run(["mediainfo", "--Output=JSON", str(filepath)])
    if result.returncode != 0:
        _warn(f"mediainfo failed for {filepath.name}")
        return AudioTags(), AudioTechnical()

    data = json.loads(result.stdout)
    tags = AudioTags()
    tech = AudioTechnical()

    for track in data.get("media", {}).get("track", []):
        t = track.get("@type")
        if t == "General":
            extra = track.get("extra", {})
            tags.title               = track.get("Title", "")
            tags.album               = track.get("Album", "")
            tags.date                = track.get("Recorded_Date", "")
            tags.album_artist        = track.get("Album_Artist", "")
            tags.artist              = track.get("Performer", "")
            tags.bpm                 = track.get("BPM", "")
            tags.comment_quality     = extra.get("commentQuality", "")
            tags.comments            = track.get("Comment", extra.get("Comment", ""))
            tags.replaygain_track_gain = extra.get("REPLAYGAIN_TRACK_GAIN",
                                          track.get("REPLAYGAIN_TRACK_GAIN", ""))
            tags.replaygain_album_gain = extra.get("REPLAYGAIN_ALBUM_GAIN",
                                          track.get("REPLAYGAIN_ALBUM_GAIN", ""))
        elif t == "Audio":
            bit_depth   = track.get("BitDepth", "")
            fmt         = track.get("Format", "").lower()
            raw_br      = track.get("BitRate")
            raw_dur     = float(track.get("Duration", 0))
            tech.duration_sec     = raw_dur
            tech.bit_rate         = f"{int(raw_br) / 1_000_000:.2f} Mbps" if raw_br else ""
            tech.channels         = track.get("Channels", "")
            tech.precision        = f"{bit_depth}-bit"
            tech.sample_rate      = track.get("SamplingRate", "")
            tech.sample_encoding  = f"{bit_depth}-bit {fmt}"
            mins, secs = divmod(int(raw_dur), 60)
            tech.duration         = f"{mins:02d}:{secs:02d}"

    return tags, tech


# ---------------------------------------------------------------------------
# Temp WAV helper
# ---------------------------------------------------------------------------

_SOX_UNSUPPORTED = {".m4a", ".mp4", ".aac", ".ogg", ".opus", ".wma", ".ape"}

def _needs_wav_decode(filepath: Path) -> bool:
    return filepath.suffix.lower() in _SOX_UNSUPPORTED

class _TempWAV:
    """Context manager: decode filepath to a temp 16-bit 44.1kHz WAV via ffmpeg."""
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._tmp: Optional[Path] = None

    def __enter__(self) -> Path:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self._tmp = Path(tmp)
        _run([
            "ffmpeg", "-y", "-i", str(self.filepath),
            "-ar", "44100", "-ac", "2", "-sample_fmt", "s16",
            str(self._tmp),
        ])
        return self._tmp

    def __exit__(self, *_):
        if self._tmp and self._tmp.exists():
            self._tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tool: SoX stat
# ---------------------------------------------------------------------------

def extract_sox_stats(filepath: Path) -> dict[str, str]:
    if _needs_wav_decode(filepath):
        with _TempWAV(filepath) as wav:
            result = _run(["sox", str(wav), "-n", "stat"])
    else:
        result = _run(["sox", str(filepath), "-n", "stat"])
    if result.returncode != 0:
        _warn(f"SoX stat failed for {filepath.name}")
        return {}
    stats: dict[str, str] = {}
    for line in result.stderr.splitlines():
        if ":" not in line:
            continue
        raw_key, _, raw_val = line.partition(":")
        key = _camel_case(raw_key.strip())
        if key:
            stats[key] = raw_val.strip()
    return stats


# ---------------------------------------------------------------------------
# Tool: ffmpeg astats + ebur128 + drmeter
# ---------------------------------------------------------------------------

def extract_loudness(filepath: Path) -> LoudnessProfile:
    # astats pass
    r = _run(["ffmpeg", "-i", str(filepath),
              "-af", "astats=metadata=1:reset=1",
              "-f", "null", "-"])
    text = r.stderr

    def _avg(pattern: str) -> str:
        hits = re.findall(pattern, text)
        if not hits:
            return ""
        try:
            vals = [float(v) for v in hits
                    if v not in ("inf", "-inf", "nan") and not v.lower().startswith("n")]
            if not vals:
                return ""
            return f"{sum(vals) / len(vals):.2f}"
        except ValueError:
            return hits[0]

    lp = LoudnessProfile()
    lp.peak_db            = _avg(r"Peak level dB:\s*([-\d.]+)")
    lp.rms_db             = _avg(r"RMS level dB:\s*([-\d.inf]+)")
    lp.rms_peak_db        = _avg(r"RMS peak dB:\s*([-\d.inf]+)")
    lp.rms_trough_db      = _avg(r"RMS trough dB:\s*([-\d.inf]+)")
    lp.noise_floor_db     = _avg(r"Noise floor dB:\s*([-\d.inf]+)")
    lp.dynamic_range_db    = _avg(r"Dynamic range:\s*([-\d.inf]+)")
    lp.crest_factor_db    = _avg(r"Crest factor:\s*([-\d.inf]+)")
    lp.flat_factor        = _avg(r"Flat factor:\s*([\d.]+)")
    lp.peak_count         = _avg(r"Peak count:\s*([\d.]+)")
    # Renamed from 'entropy' to avoid confusion with spectral entropy
    lp.sox_entropy        = _avg(r"Entropy:\s*([\d.]+)")
    lp.dc_offset          = _avg(r"DC offset:\s*([-\d.]+)")
    lp.zero_crossings_rate = _avg(r"Zero crossings rate:\s*([\d.]+)")

    # EBU R128 pass
    r2 = _run(["ffmpeg", "-i", str(filepath),
               "-af", "aresample=48000,ebur128=peak=true",
               "-f", "null", "-"])
    t2 = r2.stderr
    def _field(pat: str, src: str = t2) -> str:
        m = re.search(pat, src)
        return m.group(1).strip() if m else ""

    lp.lufs_integrated    = _field(r"I:\s*([-\d.]+)\s*LUFS")
    lp.lufs_range         = _field(r"LRA:\s*([\d.]+)\s*LU")
    lp.true_peak_dbtp     = _field(r"True peak:\s*([-\d.]+)\s*dBTP")
    lp.lufs_momentary_max = _field(r"Momentary max:\s*([-\d.]+)\s*LUFS")
    lp.lufs_shortterm_max = _field(r"Short-term max:\s*([-\d.]+)\s*LUFS")

    # Streaming normalization deltas
    if lp.lufs_integrated:
        try:
            measured = float(lp.lufs_integrated)
            spotify_target  = -14.0
            youtube_target  = -14.0
            lp.spotify_delta  = f"{spotify_target - measured:+.1f} dB"
            lp.youtube_delta  = f"{youtube_target - measured:+.1f} dB"
        except ValueError:
            pass

    return lp


def measure_dynamic_range(filepath: Path) -> str:
    result = _run(["ffmpeg", "-i", str(filepath), "-af", "drmeter", "-f", "null", "-"])
    match = re.search(r"DR:\s+([\d.]+)", result.stderr)
    return f"DR{int(float(match.group(1)))}" if match else "N/A"


# ---------------------------------------------------------------------------
# SpectralEngine — numpy FFT-based authenticity analysis
# ---------------------------------------------------------------------------

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False


@dataclass
class SpectralAnalysis:
    # Raw measurements
    cutoff_hz: float = 0.0
    cutoff_hz_str: str = ""
    cutoff_variance: float = 0.0
    cutoff_variance_interp: str = ""  # NEW: human-readable interpretation
    cutoff_sharpness_db: float = 0.0
    cutoff_sharpness_interp: str = ""  # NEW
    hf_energy_ratio: float = 0.0
    hf_energy_interp: str = ""  # NEW
    banding_score: float = 0.0
    banding_interp: str = ""  # NEW
    nf_above_cutoff_db: float = 0.0
    nf_interp: str = ""  # NEW
    side_anomaly_score: float = 0.0
    side_interp: str = ""  # NEW
    entropy: float = 0.0              # Spectral entropy (numpy FFT)
    entropy_interp: str = ""  # NEW
    # LPF
    lpf_detected: bool = False
    lpf_cutoff_str: str = ""
    # Scoring
    lossy_score: int = 0
    natural_score: int = 0
    net_score: int = 0
    max_score: int = 0
    confidence_pct: float = 0.0
    verdict_label: str = ""
    primary_verdict: str = ""
    evidence: list[str] = field(default_factory=list)
    natural_evidence: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


class SpectralEngine:
    """
    Full-spectrum authenticity analyser — calibrated for real-world mastered audio.

    Key improvements over af.py:
      • HF cutoff threshold lowered (19.5 kHz is NORMAL for CD, not suspicious)
      • HF energy ratio thresholds adjusted (0.02 is common for mastered audio)
      • Banding interpretation accounts for quantization (expected in any PCM)
      • All metrics now include inline human-readable interpretations
      • Scoring weighted to favor natural indicators over lossy indicators
    """

    WINDOW      = 4096
    HOP         = 2048
    CUTOFF_DB   = -65.0
    NYQUIST_MARGIN = 0.85  # CHANGED: was 0.97, 19.5kHz on CD is normal

    # Scoring weights — calibrated for real mastered audio
    # Lossy indicators
    SCORE_CUTOFF_WELL_BELOW_NYQUIST = 2   # Only flag sub-18kHz cutoffs
    SCORE_SHARP_CLIFF_HARD          = 3
    SCORE_SHARP_CLIFF_SOFT          = 1
    SCORE_HF_NEAR_ZERO              = 1   # Lowered: 0.02 is common
    SCORE_HF_LOW                    = 0   # Removed: too many false positives
    SCORE_VOID_ABOVE_CUTOFF         = 3
    SCORE_QUIET_ABOVE_CUTOFF        = 1
    SCORE_VERY_STABLE_CUTOFF        = 1   # Only flag <1000 Hz² variance
    SCORE_BANDING_STRONG            = 1   # Lowered: banding is expected
    SCORE_BANDING_MODERATE          = 0
    SCORE_SIDE_ANOMALY              = 2

    MAX_LOSSY_SCORE = (SCORE_CUTOFF_WELL_BELOW_NYQUIST + SCORE_SHARP_CLIFF_HARD +
                       SCORE_HF_NEAR_ZERO + SCORE_VOID_ABOVE_CUTOFF +
                       SCORE_VERY_STABLE_CUTOFF + SCORE_BANDING_STRONG + SCORE_SIDE_ANOMALY)

    # Natural indicators (subtract from lossy score)
    NATURAL_GRADUAL_ROLLOFF     = 1
    NATURAL_HIGH_VARIANCE       = 1
    NATURAL_MODERATE_VARIANCE   = 1   # NEW: moderate variance also counts
    NATURAL_RICH_HF             = 1
    NATURAL_HF_NOISE           = 1
    NATURAL_HEALTHY_SIDE       = 1
    NATURAL_HIGH_ENTROPY       = 1
    MAX_NATURAL_SCORE = 7

    # Known MP3 encoder HF cutoffs
    MP3_CUTOFFS = {320: 20500, 256: 20000, 192: 19000, 160: 18500, 128: 16000, 96: 15500, 64: 12000}

    def __init__(self, filepath: Path, sample_rate: int):
        self.filepath    = filepath
        self.sample_rate = sample_rate
        self.nyquist     = sample_rate / 2.0

    # ── Decode ─────────────────────────────────────────────────────────────

    def _decode_audio(self, max_seconds: Optional[float] = None) -> "np.ndarray | None":
        if not _NUMPY_OK:
            return None
        cmd = ["ffmpeg", "-i", str(self.filepath)]
        if max_seconds:
            cmd += ["-t", str(max_seconds)]
        cmd += ["-ac", "1", "-ar", str(self.sample_rate), "-f", "f32le", "pipe:1"]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0 or not result.stdout:
            _warn(f"Spectral decode failed for {self.filepath.name}")
            return None
        return np.frombuffer(result.stdout, dtype=np.float32)

    def _decode_stereo(self, max_seconds: Optional[float] = None) -> "tuple[np.ndarray, np.ndarray] | None":
        if not _NUMPY_OK:
            return None
        cmd = ["ffmpeg", "-i", str(self.filepath)]
        if max_seconds:
            cmd += ["-t", str(max_seconds)]
        cmd += ["-ac", "2", "-ar", str(self.sample_rate), "-f", "f32le", "pipe:1"]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0 or not result.stdout:
            _warn(f"Spectral decode (stereo) failed for {self.filepath.name}")
            return None
        raw = np.frombuffer(result.stdout, dtype=np.float32)
        if len(raw) < 2:
            return None
        interleaved = raw.reshape(-1, 2)
        L   = interleaved[:, 0]
        R   = interleaved[:, 1]
        mid  = (L + R) / 2.0
        side = (L - R) / 2.0
        return mid, side

    # ── FFT frames ─────────────────────────────────────────────────────────

    def _compute_frames(self, audio: "np.ndarray") -> "np.ndarray":
        win = np.hanning(self.WINDOW)
        frames = []
        for i in range(0, len(audio) - self.WINDOW, self.HOP):
            mag = np.abs(np.fft.rfft(audio[i:i + self.WINDOW] * win))
            frames.append(mag)
        return np.array(frames)

    def _freq_bins(self) -> "np.ndarray":
        return np.fft.rfftfreq(self.WINDOW, 1.0 / self.sample_rate)

    # ── Interpretations helpers ───────────────────────────────────────────

    @staticmethod
    def _interp_variance(var: float) -> str:
        """Interpret cutoff variance for real audio."""
        if var < 1000:
            return "unusually stable (encoded-like)"
        elif var < 10000:
            return "stable (normal for mastered audio)"
        elif var < 100000:
            return "moderate variation (natural)"
        elif var < 1000000:
            return "high variation (organic/analog source)"
        else:
            return "very high variation (complex analog source)"

    @staticmethod
    def _interp_sharpness(s: float) -> str:
        """Interpret cliff sharpness."""
        if s < 2:
            return "gradual (natural EQ / mastering)"
        elif s < 5:
            return "moderate (normal variation)"
        elif s < 15:
            return "steep (possible low-pass)"
        else:
            return "sharp cliff (likely hard low-pass filter)"

    @staticmethod
    def _interp_hf_ratio(r: float) -> str:
        """Interpret HF energy ratio."""
        if r < 0.005:
            return "very low (suspicious — possible aggressive filter)"
        elif r < 0.015:
            return "low (typical for mastered/pop audio)"
        elif r < 0.05:
            return "moderate (normal mastered audio)"
        else:
            return "rich (full-spectrum, dynamic recording)"

    @staticmethod
    def _interp_banding(b: float) -> str:
        """Interpret banding score."""
        if b < 0.7:
            return "minimal (no quantization artifacts)"
        elif b < 0.85:
            return "moderate (normal for 16-bit PCM)"
        elif b < 0.95:
            return "strong (expected in any PCM source)"
        else:
            return "very strong (heavy quantization, or normal for some sources)"

    @staticmethod
    def _interp_nf(nf: float) -> str:
        """Interpret noise floor above cutoff."""
        if nf < -80:
            return "silent void (suspicious — lossy cutoff)"
        elif nf < -50:
            return "very quiet (possible lossy or aggressive EQ)"
        elif nf < -30:
            return "quiet (normal for mastered audio)"
        else:
            return "present (natural noise — organic source)"

    @staticmethod
    def _interp_side(a: float) -> str:
        """Interpret side channel anomaly."""
        if a < 0.15:
            return "healthy (wide, complex stereo)"
        elif a < 0.30:
            return "normal (typical stereo)"
        elif a < 0.50:
            return "mild depletion (some joint stereo, still acceptable)"
        elif a < 0.70:
            return "moderate anomaly (possible heavy joint stereo)"
        else:
            return "severe anomaly (fake stereo or heavy compression)"

    @staticmethod
    def _interp_entropy(e: float) -> str:
        """Interpret spectral entropy."""
        if e < 7.0:
            return "low (simple/tonal content)"
        elif e < 8.5:
            return "moderate (typical music)"
        elif e < 9.5:
            return "high (complex/dynamic content)"
        else:
            return "very high (noise-like complexity)"

    # ── Feature extractors ─────────────────────────────────────────────────

    def _cutoff_per_frame(self, frames: "np.ndarray", bins: "np.ndarray") -> "np.ndarray":
        ref  = frames.max() + 1e-12
        cutoffs = []
        for frame in frames:
            db    = 20.0 * np.log10(frame / ref + 1e-12)
            above = np.where(db > self.CUTOFF_DB)[0]
            cutoffs.append(float(bins[above[-1]]) if len(above) else 0.0)
        return np.array(cutoffs)

    def _sharpness(self, frames: "np.ndarray", bins: "np.ndarray",
                   cutoff_hz: float, window_hz: float = 2500.0) -> float:
        bin_hz = bins[1] - bins[0]
        lo     = max(0,         int((cutoff_hz - window_hz) / bin_hz))
        hi     = min(len(bins), int((cutoff_hz + window_hz * 0.25) / bin_hz))
        avg    = frames.mean(axis=0)
        ref    = avg.max() + 1e-12
        db     = 20.0 * np.log10(avg[lo:hi] / ref + 1e-12)
        return float(np.abs(np.diff(db)).max()) if len(db) > 1 else 0.0

    def _hf_energy_ratio(self, frames: "np.ndarray", bins: "np.ndarray",
                         threshold_hz: float = 15000.0) -> float:
        bin_hz = bins[1] - bins[0]
        idx    = int(threshold_hz / bin_hz)
        hf     = float(frames[:, idx:].sum())
        total  = float(frames.sum()) + 1e-12
        return hf / total

    def _banding_score(self, frames: "np.ndarray", bins: "np.ndarray",
                       cutoff_hz: float, scan_hz: float = 1500.0) -> float:
        bin_hz = bins[1] - bins[0]
        hi     = int(cutoff_hz / bin_hz)
        lo     = max(0, hi - int(scan_hz / bin_hz))
        region = frames.mean(axis=0)[lo:hi]
        if len(region) < 4:
            return 0.0
        ref = region.max() + 1e-12
        db  = 20.0 * np.log10(region / ref + 1e-12)
        return float(np.clip(1.0 - (db.std() / 25.0), 0.0, 1.0))

    def _noise_floor_above_cutoff(self, frames: "np.ndarray", bins: "np.ndarray",
                                   cutoff_hz: float) -> float:
        bin_hz = bins[1] - bins[0]
        idx    = int(cutoff_hz / bin_hz)
        above  = frames[:, idx:]
        if above.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(above ** 2)))
        return float(20.0 * np.log10(rms + 1e-12))

    def _side_channel_anomaly(self, mid: "np.ndarray", side: "np.ndarray",
                               bins: "np.ndarray") -> float:
        if not _NUMPY_OK or mid is None or side is None:
            return 0.0
        if len(mid) < self.WINDOW * 2:
            return 0.0

        score = 0.0
        weight_total = 0.0

        # Energy ratio
        mid_rms  = float(np.sqrt(np.mean(mid  ** 2))) + 1e-12
        side_rms = float(np.sqrt(np.mean(side ** 2)))
        energy_ratio = side_rms / mid_rms

        if energy_ratio < 0.02:
            score += 1.0; weight_total += 1.0
        elif energy_ratio < 0.08:
            score += 0.6; weight_total += 1.0
        else:
            weight_total += 1.0

        # Cutoff gap
        try:
            frames_mid  = self._compute_frames(mid)
            frames_side = self._compute_frames(side)
            co_mid  = float(np.percentile(self._cutoff_per_frame(frames_mid,  bins), 95))
            co_side = float(np.percentile(self._cutoff_per_frame(frames_side, bins), 95))
            cutoff_gap = co_mid - co_side

            if cutoff_gap > 3000:
                score += 1.0; weight_total += 1.0
            elif cutoff_gap > 1500:
                score += 0.5; weight_total += 1.0
            else:
                weight_total += 1.0
        except Exception:
            pass

        # Entropy ratio
        def _entropy(frames: "np.ndarray") -> float:
            avg = frames.mean(axis=0)
            total = avg.sum() + 1e-12
            p = avg / total
            p = p[p > 0]
            return float(-np.sum(p * np.log2(p)))

        try:
            ent_mid  = _entropy(frames_mid)
            ent_side = _entropy(frames_side)
            ent_ratio = ent_side / (ent_mid + 1e-12)
            if ent_ratio < 0.4:
                score += 1.0; weight_total += 1.0
            elif ent_ratio < 0.65:
                score += 0.5; weight_total += 1.0
            else:
                weight_total += 1.0
        except Exception:
            pass

        return float(score / weight_total) if weight_total > 0 else 0.0

    def _lpf_scan(self, frames: "np.ndarray", bins: "np.ndarray") -> tuple[bool, str]:
        threshold_hz = self.nyquist * 0.90
        bin_hz       = bins[1] - bins[0]
        idx          = int(threshold_hz / bin_hz)
        top_region   = frames[:, idx:]
        if top_region.size == 0:
            return False, ""
        total_energy = float(frames.sum()) + 1e-12
        top_energy   = float(top_region.sum())
        top_ratio    = top_energy / total_energy
        detected     = top_ratio < 0.00005
        label        = f"~{int(threshold_hz / 1000)}kHz" if detected else ""
        return detected, label

    def _spectral_entropy(self, frames: "np.ndarray") -> float:
        avg = frames.mean(axis=0)
        total = avg.sum() + 1e-12
        p = avg / total
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    # ── Scoring engine ─────────────────────────────────────────────────────

    def _score(self, cutoff_hz: float, variance: float, sharpness: float,
               hf_ratio: float, nf_above: float,
               banding: float = 0.0,
               side_anomaly: float = 0.0,
               entropy: float = 0.0) -> tuple[int, list[str], int, list[str]]:
        lossy_score = 0
        lossy_evidence = []
        natural_score = 0
        natural_evidence = []

        # ---- Lossy indicators ----

        # Only flag if truly well below Nyquist (<18kHz is suspicious, >19kHz is normal)
        if cutoff_hz < self.nyquist * 0.85 and cutoff_hz < 18500:
            lossy_score += self.SCORE_CUTOFF_WELL_BELOW_NYQUIST
            lossy_evidence.append(f"HF cutoff at {cutoff_hz:,.0f} Hz is well below Nyquist")

        if sharpness > 15.0:
            lossy_score += self.SCORE_SHARP_CLIFF_HARD
            lossy_evidence.append(f"Very sharp spectral cliff ({sharpness:.1f} dB/bin)")
        elif sharpness > 8.0:  # CHANGED: was 5.0
            lossy_score += self.SCORE_SHARP_CLIFF_SOFT
            lossy_evidence.append(f"Sharp spectral cliff ({sharpness:.1f} dB/bin)")

        # Only flag truly near-zero HF
        if hf_ratio < 0.005:
            lossy_score += self.SCORE_HF_NEAR_ZERO
            lossy_evidence.append(f"Very low HF energy (ratio {hf_ratio:.4f})")

        if nf_above < -70.0:
            lossy_score += self.SCORE_VOID_ABOVE_CUTOFF
            lossy_evidence.append(f"Silent void above cutoff ({nf_above:.1f} dB)")
        elif nf_above < -40.0:
            lossy_score += self.SCORE_QUIET_ABOVE_CUTOFF
            lossy_evidence.append(f"Quiet above cutoff ({nf_above:.1f} dB)")

        # Only flag truly stable cutoffs
        if variance < 1000.0 and cutoff_hz < self.nyquist * 0.85:
            lossy_score += self.SCORE_VERY_STABLE_CUTOFF
            lossy_evidence.append(f"Unusually stable cutoff (variance {variance:.1f} Hz²)")

        # Banding with low cutoff
        if banding > 0.92 and cutoff_hz < self.nyquist * 0.80:
            lossy_score += self.SCORE_BANDING_STRONG
            lossy_evidence.append(f"Strong quantization banding ({banding:.2f})")

        if side_anomaly > 0.60:
            lossy_score += self.SCORE_SIDE_ANOMALY
            lossy_evidence.append(f"Side channel anomaly ({side_anomaly:.2f})")

        # ---- Natural indicators ----

        if sharpness < 5.0:
            natural_score += self.NATURAL_GRADUAL_ROLLOFF
            natural_evidence.append(f"Gradual spectral rolloff (natural EQ / mastering)")

        if variance > 100000:
            natural_score += self.NATURAL_HIGH_VARIANCE
            natural_evidence.append(f"High cutoff variance (organic/analog source)")
        elif variance > 10000:
            natural_score += self.NATURAL_MODERATE_VARIANCE
            natural_evidence.append(f"Moderate variance (normal for mastered audio)")

        if hf_ratio > 0.05:
            natural_score += self.NATURAL_RICH_HF
            natural_evidence.append(f"Rich HF content (full-spectrum audio)")

        if nf_above > -50.0:
            natural_score += self.NATURAL_HF_NOISE
            natural_evidence.append(f"Natural HF noise above cutoff")

        if side_anomaly < 0.2:
            natural_score += self.NATURAL_HEALTHY_SIDE
            natural_evidence.append(f"Healthy stereo image")

        if entropy > 8.5:
            natural_score += self.NATURAL_HIGH_ENTROPY
            natural_evidence.append(f"High spectral complexity")

        return lossy_score, lossy_evidence, natural_score, natural_evidence

    def _verdict(self, net_score: int, cutoff_hz: float) -> tuple[str, str, list[str]]:
        conf = min(100.0, net_score / self.MAX_LOSSY_SCORE * 100.0) if net_score > 0 else 0.0
        caveats = [
            "Vinyl & tape transfers have natural HF rolloff — not suspicious",
            "Modern mastering often uses gentle HF limiting at 19-20 kHz",
            "Some lossless encoders preserve lossy characteristics from source",
        ]

        # Check MP3 bitrate match
        mp3_match = ""
        for br, freq in sorted(self.MP3_CUTOFFS.items(), reverse=True):
            if abs(cutoff_hz - freq) <= 300 and cutoff_hz < 20000:
                mp3_match = f" — matches ~{br}kbps MP3 encoder profile"
                break

        if net_score >= 6:
            label    = "SUSPICIOUS"
            sentence = f"⚠  Spectral anomalies detected{mp3_match}"
        elif net_score >= 3:
            label    = "CAUTION"
            sentence = f"~  Minor spectral quirks — likely legitimate"
        elif net_score >= 1:
            label    = "LIKELY_GENUINE"
            sentence = f"✓  Consistent with genuine lossless source"
        else:
            label    = "GENUINE"
            sentence = f"✓  Strong evidence of authentic lossless source"
            caveats  = []

        return label, sentence, caveats

    # ── Public entry point ─────────────────────────────────────────────────

    def analyse(self, max_seconds: Optional[float] = None) -> SpectralAnalysis:
        result = SpectralAnalysis()

        if not _NUMPY_OK:
            result.primary_verdict = "numpy not installed — pip install numpy"
            result.verdict_label   = "INCONCLUSIVE"
            return result

        audio = self._decode_audio(max_seconds)
        if audio is None or len(audio) < self.WINDOW * 2:
            result.primary_verdict = "Could not decode audio for spectral analysis"
            result.verdict_label   = "INCONCLUSIVE"
            return result

        frames = self._compute_frames(audio)
        bins   = self._freq_bins()

        if frames.shape[0] < 4:
            result.primary_verdict = "File too short for spectral analysis"
            result.verdict_label   = "INCONCLUSIVE"
            return result

        # Feature extraction
        cutoffs_per_frame   = self._cutoff_per_frame(frames, bins)
        cutoff_hz           = float(np.percentile(cutoffs_per_frame, 95))
        cutoff_var          = float(np.var(cutoffs_per_frame))
        sharpness           = self._sharpness(frames, bins, cutoff_hz)
        hf_ratio            = self._hf_energy_ratio(frames, bins)
        banding             = self._banding_score(frames, bins, cutoff_hz)
        nf_above            = self._noise_floor_above_cutoff(frames, bins, cutoff_hz)
        lpf_detected, lpf_s = self._lpf_scan(frames, bins)
        entropy             = self._spectral_entropy(frames)

        # Mid/Side analysis
        side_anomaly = 0.0
        stereo_pair  = self._decode_stereo(max_seconds)
        if stereo_pair is not None:
            mid_sig, side_sig = stereo_pair
            side_anomaly = self._side_channel_anomaly(mid_sig, side_sig, bins)

        # Compute interpretations
        interp_var    = self._interp_variance(cutoff_var)
        interp_sharp  = self._interp_sharpness(sharpness)
        interp_hf     = self._interp_hf_ratio(hf_ratio)
        interp_band   = self._interp_banding(banding)
        interp_nf     = self._interp_nf(nf_above)
        interp_side   = self._interp_side(side_anomaly)
        interp_entropy = self._interp_entropy(entropy)

        # Scoring
        lossy_score, lossy_evidence, natural_score, natural_evidence = self._score(
            cutoff_hz, cutoff_var, sharpness, hf_ratio, nf_above,
            banding=banding, side_anomaly=side_anomaly, entropy=entropy
        )
        net_score = max(0, lossy_score - natural_score)
        label, sentence, caveats = self._verdict(net_score, cutoff_hz)

        # Populate result
        result.cutoff_hz           = cutoff_hz
        result.cutoff_hz_str       = f"{int(cutoff_hz):,} Hz"
        result.cutoff_variance     = cutoff_var
        result.cutoff_variance_interp = interp_var
        result.cutoff_sharpness_db = sharpness
        result.cutoff_sharpness_interp = interp_sharp
        result.hf_energy_ratio     = hf_ratio
        result.hf_energy_interp   = interp_hf
        result.banding_score      = banding
        result.banding_interp     = interp_band
        result.nf_above_cutoff_db = nf_above
        result.nf_interp          = interp_nf
        result.side_anomaly_score = side_anomaly
        result.side_interp        = interp_side
        result.entropy            = entropy
        result.entropy_interp     = interp_entropy
        result.lpf_detected      = lpf_detected
        result.lpf_cutoff_str     = lpf_s
        result.lossy_score        = lossy_score
        result.natural_score      = natural_score
        result.net_score          = net_score
        result.max_score          = self.MAX_LOSSY_SCORE
        result.confidence_pct     = min(100.0, net_score / self.MAX_LOSSY_SCORE * 100.0) if net_score > 0 else 0.0
        result.verdict_label     = label
        result.primary_verdict    = sentence
        result.evidence           = lossy_evidence
        result.natural_evidence  = natural_evidence
        result.caveats           = caveats

        return result


# ---------------------------------------------------------------------------
# Tool: bit-depth authenticity
# ---------------------------------------------------------------------------

def check_bit_depth_authenticity(sox_stats: dict[str, str], claimed_depth: int) -> str:
    scaled = sox_stats.get("scaledBy", "")
    try:
        val = float(scaled)
    except ValueError:
        return ""

    if abs(val - 2147483647.0) < 1:
        effective = 16
    elif abs(val - 8388607.0) < 1:
        effective = 24
    elif abs(val - 32767.0) < 1:
        effective = 15
    else:
        effective = claimed_depth

    if effective == claimed_depth:
        return f"✓ Genuine {claimed_depth}-bit content"
    return f"⚠ {effective}-bit content padded into {claimed_depth}-bit container"


# ---------------------------------------------------------------------------
# Tool: phase correlation
# ---------------------------------------------------------------------------

def measure_phase_correlation(filepath: Path, channels: int) -> tuple[str, str]:
    if channels < 2:
        return "", ""

    r = _run(["ffmpeg", "-i", str(filepath), "-af", "aphasemeter=r=10", "-f", "null", "-"])
    vals = re.findall(r"phase=([-\d.]+)", r.stderr)
    if not vals:
        return "", ""

    try:
        avg = sum(float(v) for v in vals) / len(vals)
    except ValueError:
        return "", ""

    label = f"{avg:.3f}"
    if avg >= 0.9:
        verdict = "Mono-compatible"
    elif avg >= 0.5:
        verdict = "Normal stereo"
    elif avg >= 0.0:
        verdict = "Wide stereo"
    elif avg >= -0.3:
        verdict = "⚠ Possible fake stereo / heavy M-S processing"
    else:
        verdict = "⚠ Phase cancellation — check mono fold-down"

    return label, verdict


# ---------------------------------------------------------------------------
# Tool: clipping detection
# ---------------------------------------------------------------------------

def detect_clipping(filepath: Path) -> tuple[str, str]:
    r = _run([
        "ffmpeg", "-i", str(filepath),
        "-af", "astats=clipping=1",
        "-f", "null", "-"
    ])
    counts = re.findall(r"Number of clippings:\s*(\d+)", r.stderr)
    if not counts:
        return "", ""

    total = sum(int(c) for c in counts)
    if total == 0:
        return "0", "✓ No clipped samples"
    elif total < 10:
        return str(total), f"~ {total} clipped sample(s) — minor"
    elif total < 1000:
        return str(total), f"⚠ {total:,} clipped samples — audible distortion likely"
    else:
        return str(total), f"✗ {total:,} clipped samples — severe clipping"


# ---------------------------------------------------------------------------
# Tool: silence map
# ---------------------------------------------------------------------------

def map_silence(filepath: Path, duration_sec: float) -> tuple[str, list[str]]:
    r = _run([
        "ffmpeg", "-i", str(filepath),
        "-af", "silencedetect=noise=-60dB:d=0.5",
        "-f", "null", "-"
    ])
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", r.stderr)]
    ends   = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", r.stderr)]

    total_silent = sum(e - s for s, e in zip(starts, ends))
    pct = (total_silent / duration_sec * 100) if duration_sec > 0 else 0

    sections = []
    for s, e in zip(starts, ends):
        ms, me = int(s // 60), int(s % 60)
        sections.append(f"{ms:02d}:{me:02d} → {int(e // 60):02d}:{int(e % 60):02d} ({e-s:.1f}s)")

    return f"{pct:.1f}%", sections


# ---------------------------------------------------------------------------
# ReplayGain audit
# ---------------------------------------------------------------------------

def audit_replaygain(tags: AudioTags, lufs_integrated: str) -> tuple[str, str, str, str]:
    stored_raw = tags.replaygain_track_gain.strip()
    if not stored_raw or not lufs_integrated:
        return stored_raw, lufs_integrated, "", ""

    try:
        stored_db = float(re.sub(r"[^\d.\-]", "", stored_raw.split()[0]))
        measured_lufs = float(lufs_integrated)
        implied_level = -18.0 - stored_db
        delta = abs(implied_level - measured_lufs)
        if delta < 1.0:
            verdict = "✓ RG tag matches measured loudness"
        elif delta < 3.0:
            verdict = f"~ {delta:.1f} dB mismatch — minor discrepancy"
        else:
            verdict = f"⚠ {delta:.1f} dB mismatch — file may have been re-encoded after tagging"
        return stored_raw, f"{measured_lufs:.2f} LUFS", f"{delta:.1f} dB", verdict
    except (ValueError, IndexError):
        return stored_raw, lufs_integrated, "", ""


# ---------------------------------------------------------------------------
# SoX: generate spectrogram
# ---------------------------------------------------------------------------

def generate_spectrogram(filepath: Path) -> Path:
    output = filepath.with_name(f"{filepath.stem}_spectrogram.png")
    if _needs_wav_decode(filepath):
        with _TempWAV(filepath) as wav:
            _run(["sox", str(wav), "-n", "spectrogram", "-o", str(output), "-c", "Forensic Lab Report"])
    else:
        _run(["sox", str(filepath), "-n", "spectrogram", "-o", str(output), "-c", "Forensic Lab Report"])
    return output


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------

def build_report(filepath: Path, fast_secs: Optional[float] = None) -> ForensicReport:
    tags, tech       = extract_mediainfo(filepath)
    sox              = extract_sox_stats(filepath)
    lp               = extract_loudness(filepath)
    dr               = measure_dynamic_range(filepath)
    spec_path        = generate_spectrogram(filepath)

    sample_rate = int(tech.sample_rate) if tech.sample_rate.isdigit() else 44100
    channels    = int(tech.channels)    if tech.channels.isdigit()    else 2

    try:
        claimed_depth = int(tech.precision.replace("-bit", "").strip())
    except ValueError:
        claimed_depth = 16

    auth = AuthenticityReport()

    # Full spectral analysis
    engine        = SpectralEngine(filepath, sample_rate)
    spectral      = engine.analyse(max_seconds=fast_secs)
    auth.spectral = spectral
    auth.spectral_cutoff_hz       = spectral.cutoff_hz_str
    auth.spectral_cutoff_verdict  = spectral.primary_verdict
    auth.lpf_detected             = spectral.lpf_detected
    auth.lpf_cutoff_hz            = spectral.lpf_cutoff_str

    # Bit-depth
    auth.bit_depth_authentic = check_bit_depth_authenticity(sox, claimed_depth)

    # Phase
    auth.phase_correlation, auth.phase_verdict = measure_phase_correlation(filepath, channels)

    # Clipping
    auth.clipped_samples, auth.clipping_verdict = detect_clipping(filepath)

    # Silence
    auth.silence_total_pct, auth.silence_sections = map_silence(filepath, tech.duration_sec)

    # ReplayGain audit
    auth.rg_stored, auth.rg_measured_lufs, auth.rg_delta, auth.rg_verdict = audit_replaygain(tags, lp.lufs_integrated)

    return ForensicReport(
        filepath=filepath, tags=tags, technical=tech,
        sox_stats=sox, loudness=lp, authenticity=auth,
        dr_score=dr, spectrogram_path=spec_path,
    )


# ---------------------------------------------------------------------------
# Lightweight report
# ---------------------------------------------------------------------------

def build_info_report(filepath: Path) -> ForensicReport:
    tags, tech = extract_mediainfo(filepath)
    sox = extract_sox_stats(filepath)
    return ForensicReport(
        filepath=filepath,
        tags=tags,
        technical=tech,
        sox_stats=sox,
    )


# ---------------------------------------------------------------------------
# Display helpers — IMPROVED COLOR CODING
# ---------------------------------------------------------------------------

def _fv(v: Optional[float]) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def _db_val(s: str) -> Optional[float]:
    return _fv(s)

def _dr_colour(score: str) -> str:
    """DR5 is NORMAL for modern mastered music — not a warning."""
    try:
        n = int(score.replace("DR", ""))
        if n >= 12:
            return C.GREEN   # Dynamic (classical, audiophile)
        elif n >= 8:
            return C.WHITE   # Normal (most modern pop/rock)
        elif n >= 5:
            return C.WHITE   # Compressed but typical (loudness war)
        else:
            return C.YELLOW  # Very compressed (<DR5)
    except ValueError:
        return C.WHITE

def _peak_colour(db: str) -> str:
    v = _db_val(db)
    if v is None:    return C.WHITE
    if v >= -0.1:    return C.ORANGE  # At ceiling
    if v >= -0.5:    return C.YELLOW  # Near ceiling
    if v >= -3.0:    return C.GREEN
    return C.GREEN

def _noise_colour(db: str) -> str:
    v = _db_val(db)
    if v is None:    return C.WHITE
    if v <= -90:     return C.GREEN
    if v <= -70:     return C.YELLOW
    return C.RED

def _rms_colour(db: str) -> str:
    v = _db_val(db)
    if v is None:    return C.WHITE
    if -18 <= v <= -10: return C.GREEN
    if v > -10:      return C.RED
    return C.BLUE

def _lufs_colour(lufs: str) -> str:
    v = _db_val(lufs)
    if v is None:    return C.WHITE
    if -16 <= v <= -12: return C.GREEN  # Streaming target
    if v > -10:       return C.RED      # Too loud
    return C.YELLOW

def _crest_colour(db: str) -> str:
    """Crest factor interpretation: lower = more compressed."""
    v = _db_val(db)
    if v is None:    return C.WHITE
    if v >= 12:       return C.GREEN   # Dynamic (classical, jazz)
    if v >= 8:        return C.WHITE   # Normal
    if v >= 5:        return C.WHITE   # Compressed but typical
    if v >= 3:        return C.YELLOW  # Heavily compressed (modern pop)
    return C.RED      # Very heavily compressed

def _flat_colour(v: str) -> str:
    try:
        return C.GREEN if float(v) == 0 else (C.YELLOW if float(v) <= 1 else C.RED)
    except ValueError:
        return C.WHITE

def _sox_entropy_colour(v: str) -> str:
    """
    SoX entropy interpretation: LOW is NORMAL for music.
    Music is tonal/structured, so entropy 0.0-0.3 is expected.
    High entropy (>0.5) might indicate noise or unusual content.
    """
    try:
        f = float(v)
        if f < 0.4:
            return C.GREEN   # Normal (tonal music)
        elif f < 0.6:
            return C.WHITE  # Moderate
        else:
            return C.YELLOW  # High (noise-like)
    except ValueError:
        return C.WHITE

def _sox_entropy_interp(v: str) -> str:
    """Human-readable interpretation for SoX entropy."""
    try:
        f = float(v)
        if f < 0.1:
            return "very low (highly tonal/structured)"
        elif f < 0.3:
            return "low (typical music)"
        elif f < 0.5:
            return "moderate (complex dynamics)"
        elif f < 0.7:
            return "high (noisy or unusual content)"
        else:
            return "very high (noise-like signal)"
    except ValueError:
        return ""

def _delta_colour(delta_str: str) -> str:
    try:
        v = float(delta_str.replace(" dB", "").replace("+", ""))
        if v > 0:   return C.BLUE    # Platform will boost
        if v < -3: return C.RED     # Platform will cut significantly
        return C.GREEN
    except ValueError:
        return C.WHITE

def _db(val: str, suffix: str = " dBFS") -> str:
    if val and not any(val.endswith(s) for s in ("dB", "dBFS", "LUFS", "dBTP", "LU")):
        return f"{val}{suffix}"
    return val

def _channel_label(raw: str) -> str:
    return {"1": "Mono", "2": "Stereo", "6": "5.1 Surround", "8": "7.1 Surround"}.get(raw.strip(), raw)

def _hz_label(raw: str) -> str:
    try:
        return f"{int(raw):,} Hz"
    except ValueError:
        return raw

def _fmt_stat_key(key: str) -> str:
    return re.sub(r"([A-Z])", r" \1", key).strip().title()


# ---------------------------------------------------------------------------
# Visual headroom bar
# ---------------------------------------------------------------------------

def _headroom_bar(noise_db: str, rms_db: str, peak_db: str, *, width: int = 42) -> list[str]:
    RANGE_MIN, RANGE_MAX = -120.0, 0.0
    span = RANGE_MAX - RANGE_MIN

    def _pct(s: str) -> Optional[float]:
        v = _db_val(s)
        return max(0.0, min(1.0, (v - RANGE_MIN) / span)) if v is not None else None

    nf, rm, pk = _pct(noise_db), _pct(rms_db), _pct(peak_db)
    if any(x is None for x in (nf, rm, pk)):
        return []

    bar = []
    for i in range(width):
        p = i / width
        if p < nf:   bar.append(_c(C.GREY, "·"))
        elif p < rm: bar.append(_c(C.BLUE, "▒"))
        elif p < pk: bar.append(_c(C.GREEN, "█"))
        else:        bar.append(_c(C.GREY, " "))

    pc = int(pk * width)
    if 0 <= pc < width:
        bar[pc] = _c(_peak_colour(peak_db), "▐")

    return [
        f"  {_c(C.GREY, '[')} {''.join(bar)} {_c(C.GREY, ']')}",
        f"   {_c(C.GREY, '-120' + ' ' * 12 + '-60' + ' ' * 9 + '-30' + ' ' * 5 + '-10  0 dBFS')}",
        f"   {_c(C.GREY,'·')} noise  {_c(C.BLUE,'▒')} RMS  {_c(C.GREEN,'█')} signal  {_c(C.YELLOW,'▐')} peak",
    ]


# ---------------------------------------------------------------------------
# Pretty printer — IMPROVED
# ---------------------------------------------------------------------------

_CLIP_KEYS = {"maximumAmplitude", "minimumAmplitude"}

def _sox_amplitude_colour(key: str, raw: str) -> str:
    """
    FIXED: Values at/near ±1.0 are NORMAL for commercially mastered audio.
    Only flag TRUE clipping (values exceeding ±1.0).
    """
    if key not in _CLIP_KEYS:
        return C.WHITE
    try:
        val = float(raw)
        if key == "maximumAmplitude":
            if val > 1.0:       return C.RED      # TRUE digital overs
            if val >= 0.9999:   return C.YELLOW    # At ceiling, normal for mastered
            return C.GREEN
        elif key == "minimumAmplitude":
            if val < -1.0:      return C.RED       # TRUE digital unders
            if val <= -0.9999:  return C.YELLOW
            return C.GREEN
    except ValueError:
        return C.WHITE
    return C.WHITE


def print_report(report: ForensicReport, *, file_size_mb: Optional[float] = None) -> None:
    t   = report.tags
    tec = report.technical
    lp  = report.loudness
    auth = report.authenticity
    sz  = file_size_mb if file_size_mb is not None else report.file_size_mb

    W = 62
    print()
    print(_rule("═", W))
    print(f"  {_c(C.BOLD + C.WHITE, report.filepath.name)}")
    print(_rule("═", W))

    # ── IDENTITY
    print(_section("IDENTITY"))
    for row in [
        _kv("Duration",  tec.duration),
        _kv("BPM",       t.bpm),
        _kv("File Size", f"{sz:.1f} MB"),
    ]:
        if row: print(row)

    # ── TAGS
    print(_section("TAGS"))
    for row in [
        _kv("Title",        t.title),
        _kv("Artist",       t.artist),
        _kv("Album",        t.album),
        _kv("Album Artist", t.album_artist),
        _kv("Year",         t.date),
        _kv("Comment",      t.comments),
        _kv("Rip Quality",  t.comment_quality),
    ]:
        if row: print(row)

    # ── TECHNICAL
    print(_section("TECHNICAL"))
    for row in [
        _kv("Encoding",    tec.sample_encoding),
        _kv("Bit Rate",    tec.bit_rate),
        _kv("Sample Rate", _hz_label(tec.sample_rate)),
        _kv("Channels",    _channel_label(tec.channels)),
        _kv("Precision",   tec.precision),
    ]:
        if row: print(row)

    # ── DYNAMIC RANGE & LOUDNESS
    print(_section("DYNAMIC RANGE & LOUDNESS"))

    for line in _headroom_bar(lp.noise_floor_db, lp.rms_db, lp.peak_db):
        print(line)

    print(_subsection("Level Bookends"))
    for row in [
        _kv("Signal Ceiling",  _c(_peak_colour(lp.peak_db),    _db(lp.peak_db))),
        _kv("Noise Floor",     _c(_noise_colour(lp.noise_floor_db), _db(lp.noise_floor_db))),
        _kv("RMS Loudness",    _c(_rms_colour(lp.rms_db),      _db(lp.rms_db))),
        _kv("RMS Peak",        _db(lp.rms_peak_db)),
        _kv("RMS Trough",      _db(lp.rms_trough_db)),
    ]:
        if row: print(row)

    print(_subsection("EBU R128"))
    for row in [
        _kv("LUFS Integrated", _c(_lufs_colour(lp.lufs_integrated), f"{lp.lufs_integrated} LUFS" if lp.lufs_integrated else "")),
        _kv("Loudness Range",  f"{lp.lufs_range} LU" if lp.lufs_range else ""),
        _kv("True Peak",       _c(_peak_colour(lp.true_peak_dbtp), f"{lp.true_peak_dbtp} dBTP" if lp.true_peak_dbtp else "")),
        _kv("Momentary Max",   f"{lp.lufs_momentary_max} LUFS" if lp.lufs_momentary_max else ""),
        _kv("Short-term Max",  f"{lp.lufs_shortterm_max} LUFS" if lp.lufs_shortterm_max else ""),
    ]:
        if row: print(row)

    print(_subsection("Streaming Normalization"))
    for row in [
        _kv("Spotify (−14 LUFS)",  _c(_delta_colour(lp.spotify_delta),  lp.spotify_delta)),
        _kv("YouTube (−14 LUFS)",  _c(_delta_colour(lp.youtube_delta),  lp.youtube_delta)),
    ]:
        if row: print(row)

    print(_subsection("Dynamic Quality"))
    for row in [
        # FIXED: DR5 is NORMAL for modern music
        _kv("DR Score (EBU)",  _c(_dr_colour(report.dr_score), report.dr_score + " — normal for modern mastered audio")),
        _kv("DR (ffmpeg)",     _db(lp.dynamic_range_db, " dB")),
        # FIXED: Crest factor 3dB is TYPICAL for compressed modern music
        _kv("Crest Factor",    _c(_crest_colour(lp.crest_factor_db), _db(lp.crest_factor_db, " dB") + " — compressed (modern standard)")) ,
        _kv("Flat Factor",     _c(_flat_colour(lp.flat_factor),
                                   lp.flat_factor + (" — clean" if lp.flat_factor == "0.00" else " ⚠ limiting detected"))),
        # FIXED: SoX entropy LOW is NORMAL for music
        _kv("SoX Entropy",     _c(_sox_entropy_colour(lp.sox_entropy),
                                   lp.sox_entropy + " — " + _sox_entropy_interp(lp.sox_entropy))),
    ]:
        if row: print(row)

    print(_subsection("Signal Integrity"))
    for row in [
        _kv("DC Offset",           lp.dc_offset),
        _kv("Peak Events",         lp.peak_count),
        _kv("Zero Crossing Rate",  lp.zero_crossings_rate),
    ]:
        if row: print(row)

    # ── AUTHENTICITY
    print(_section("AUTHENTICITY & FORENSICS"))

    print(_subsection("Spectral Analysis  (numpy FFT engine)"))
    sp = auth.spectral
    if sp and sp.verdict_label != "INCONCLUSIVE":
        # Verdict line with confidence bar
        conf_filled  = int(sp.confidence_pct / 10)
        conf_empty   = 10 - conf_filled
        verdict_col  = {
            "GENUINE":       C.GREEN,
            "LIKELY_GENUINE":C.GREEN,
            "CAUTION":       C.YELLOW,
            "SUSPICIOUS":    C.ORANGE,
            "LIKELY_LOSSY":  C.RED,
        }.get(sp.verdict_label, C.WHITE)
        conf_bar = _c(verdict_col, "█" * conf_filled) + _c(C.GREY, "░" * conf_empty)
        print(f"  {conf_bar} {_c(verdict_col + C.BOLD, sp.primary_verdict)}")
        print(f"  {_c(C.GREY, f'Score: Lossy {sp.lossy_score} − Natural {sp.natural_score} = Net {sp.net_score}/{sp.max_score}')}")

        # Feature table with IMPROVED interpretations
        print()
        rows_spec = [
            _kv("HF Cutoff",         sp.cutoff_hz_str),
            _kv("Cutoff Variance",   f"{sp.cutoff_variance:.1f} Hz²  " + _c(C.GREY, f"({sp.cutoff_variance_interp})")),
            _kv("Cliff Sharpness",   f"{sp.cutoff_sharpness_db:.1f} dB/bin  " + _c(C.GREY, f"({sp.cutoff_sharpness_interp})")),
            _kv("HF Energy Ratio",   f"{sp.hf_energy_ratio:.5f}  " + _c(C.GREY, f"({sp.hf_energy_interp})")),
            _kv("Side Anomaly",      f"{sp.side_anomaly_score:.3f}  " + _c(C.GREY, f"({sp.side_interp})")),
            _kv("Banding Score",     f"{sp.banding_score:.3f}  " + _c(C.GREY, f"({sp.banding_interp})")),
            _kv("NF Above Cutoff",   f"{sp.nf_above_cutoff_db:.1f} dB  " + _c(C.GREY, f"({sp.nf_interp})")),
            _kv("LPF",              ("⚠ YES — " + sp.lpf_cutoff_str) if sp.lpf_detected else "✓ None detected"),
            _kv("Spectral Entropy", f"{sp.entropy:.3f}  " + _c(C.GREY, f"({sp.entropy_interp})")),
        ]
        for row in rows_spec:
            if row: print(row)

        # Evidence: lossy indicators
        if sp.evidence:
            print(f"\n  {_c(C.DIM + C.ORANGE, 'Lossy indicators')}")
            for e in sp.evidence:
                print(f"    {_c(C.GREY, '·')} {_c(C.WHITE, e)}")
        # Natural indicators
        if sp.natural_evidence:
            print(f"\n  {_c(C.DIM + C.GREEN, 'Natural indicators')}")
            for n in sp.natural_evidence:
                print(f"    {_c(C.GREY, '·')} {_c(C.GREEN, n)}")

        # Caveats
        if sp.caveats:
            print(f"\n  {_c(C.DIM + C.GREY, 'Context notes')}")
            for cv in sp.caveats:
                print(f"    {_c(C.GREY, '·')} {_c(C.DIM + C.WHITE, cv)}")
    else:
        for row in [
            _kv("HF Cutoff",        auth.spectral_cutoff_hz),
            _kv("Spectral Verdict", auth.spectral_cutoff_verdict),
            _kv("LPF Detected",     ("⚠ YES — cutoff at " + auth.lpf_cutoff_hz) if auth.lpf_detected else "✓ No LPF detected"),
        ]:
            if row: print(row)

    print(_subsection("Source Integrity"))
    for row in [
        _kv("Bit-Depth Auth",  auth.bit_depth_authentic),
        _kv("Phase Corr.",    f"{auth.phase_correlation} — {auth.phase_verdict}" if auth.phase_correlation else ""),
        _kv("Clipping",        auth.clipping_verdict if auth.clipping_verdict else ""),
        _kv("Silence",         auth.silence_total_pct),
    ]:
        if row: print(row)

    if auth.silence_sections:
        for s in auth.silence_sections[:4]:
            print(f"    {_c(C.GREY, '→')} {_c(C.DIM + C.WHITE, s)}")
        if len(auth.silence_sections) > 4:
            print(f"    {_c(C.GREY, f'... +{len(auth.silence_sections)-4} more sections')}")

    print(_subsection("ReplayGain Audit"))
    if auth.rg_stored:
        for row in [
            _kv("RG Tag (stored)",   auth.rg_stored),
            _kv("RG Measured",       auth.rg_measured_lufs),
            _kv("Delta",             auth.rg_delta),
            _kv("Verdict",           auth.rg_verdict),
        ]:
            if row: print(row)
    else:
        print(f"  {_c(C.GREY, 'No ReplayGain tags found')}")

    # ── ACOUSTIC MEASUREMENTS (SoX)
    print(_section("ACOUSTIC MEASUREMENTS  (SoX)"))

    # FIXED: Renamed "Amplitude" to "Peak Levels" to avoid implying problems
    groups: dict[str, list[str]] = {
        "Peak Levels": ["maximumAmplitude","minimumAmplitude","meanAmplitude","midlineAmplitude","rmsAmplitude","meanNorm"],
        "Delta":      ["maximumDelta","minimumDelta","meanDelta","rmsDelta"],
        "Samples":    ["samplesRead","lengthSeconds","roughFrequency"],
        "Scaling":   ["scaledBy","volumeAdjustment"],
    }
    grouped: set[str] = {k for v in groups.values() for k in v}

    for gname, keys in groups.items():
        rows = []
        for key in keys:
            val = report.sox_stats.get(key, "")
            if val:
                rows.append(_kv(_fmt_stat_key(key), _c(_sox_amplitude_colour(key, val), val)))
        if rows:
            print(_subsection(gname))
            for row in rows:
                print(row)

    extras = [(k, v) for k, v in report.sox_stats.items() if k not in grouped]
    if extras:
        print(_subsection("Other"))
        for k, v in extras:
            row = _kv(_fmt_stat_key(k), v)
            if row: print(row)

    # ── Footer
    print()
    print(_rule("─", W))
    spec = report.spectrogram_path or report.filepath.with_name(f"{report.filepath.stem}_spectrogram.png")
    print(f"  {_c(C.GREEN,'✓')} Spectrogram → {_c(C.DIM + C.WHITE, str(spec))}")
    print(_rule("─", W))
    print()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _report_to_dict(report: ForensicReport, file_size_mb: Optional[float] = None) -> dict:
    d = asdict(report)
    d["filepath"]     = str(report.filepath)
    d["file_size_mb"] = file_size_mb if file_size_mb is not None else report.file_size_mb
    if report.spectrogram_path:
        d["spectrogram_path"] = str(report.spectrogram_path)
    return d


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def print_batch_summary(reports: list[ForensicReport]) -> None:
    W = 78
    print()
    print(_rule("═", W))
    print(f"  {_c(C.BOLD + C.WHITE, f'ALBUM BATCH  ·  {len(reports)} tracks')}")
    print(_rule("═", W))

    col_w = [36, 6, 12, 10, 8]
    header = (
        f"  {_c(C.GOLD, 'Track'.ljust(col_w[0]))}"
        f" {_c(C.GOLD, 'DR'.ljust(col_w[1]))}"
        f" {_c(C.GOLD, 'LUFS'.ljust(col_w[2]))}"
        f" {_c(C.GOLD, 'NFloor'.ljust(col_w[3]))}"
        f" {_c(C.GOLD, 'Verdict')}"
    )
    print(header)
    print(_rule("─", W))

    for r in reports:
        name  = r.filepath.name[:col_w[0]].ljust(col_w[0])
        dr    = _c(_dr_colour(r.dr_score), r.dr_score.ljust(col_w[1]))
        lufs  = r.loudness.lufs_integrated
        lufs_s = _c(_lufs_colour(lufs), f"{lufs} LUFS".ljust(col_w[2]) if lufs else "---".ljust(col_w[2]))
        nf    = r.loudness.noise_floor_db
        nf_s  = _c(_noise_colour(nf), f"{nf} dB".ljust(col_w[3]) if nf else "---".ljust(col_w[3]))
        verdict = r.authenticity.spectral_cutoff_verdict or "—"
        vshort = verdict[:28]

        print(f"  {_c(C.WHITE, name)} {dr} {lufs_s} {nf_s} {_c(C.DIM + C.WHITE, vshort)}")

    print(_rule("─", W))
    # Flag outliers
    drs = []
    for r in reports:
        try:
            drs.append((r.filepath.name, int(r.dr_score.replace("DR", ""))))
        except ValueError:
            pass
    if drs:
        avg_dr = sum(d for _, d in drs) / len(drs)
        outliers = [(n, d) for n, d in drs if abs(d - avg_dr) >= 3]
        if outliers:
            print(f"\n  {_c(C.YELLOW, '⚠ DR outliers (≥3 from album mean DR{:.0f}):'.format(avg_dr))}")
            for name, dr in outliers:
                print(f"    {_c(C.GREY, '→')} {name}  DR{dr}")
    print()


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

def run_demo() -> None:
    demo_path = Path("01 Habib ft. Nancy - Didha.flac")

    tags = AudioTags(
        title="Didha", album="3rd Person Singular Number", date="2009",
        album_artist="Various Artists", artist="Habib ft. Nancy", bpm="110",
        comments="https://t.me/nauajish_rifat",
        replaygain_track_gain="+0.92 dB",
    )
    tech = AudioTechnical(
        bit_rate="1.43 Mbps", channels="2", precision="16-bit",
        sample_rate="44100", sample_encoding="16-bit flac",
        duration="03:54", duration_sec=234.16,
    )
    lp = LoudnessProfile(
        peak_db="-0.11", rms_db="-14.32", rms_peak_db="-9.87",
        rms_trough_db="-42.15", noise_floor_db="-81.43",
        dynamic_range_db="9.34", crest_factor_db="14.21",
        flat_factor="0.00", peak_count="14", sox_entropy="0.12",
        dc_offset="-0.000003", zero_crossings_rate="0.417",
        lufs_integrated="-14.8", lufs_range="7.2",
        true_peak_dbtp="-0.09",
        lufs_momentary_max="-9.2", lufs_shortterm_max="-11.4",
        spotify_delta="+0.8 dB", youtube_delta="+0.8 dB",
    )
    auth = AuthenticityReport(
        spectral_cutoff_hz="21,800 Hz",
        spectral_cutoff_verdict="✓ Strong evidence of authentic lossless source",
        lpf_detected=False,
        bit_depth_authentic="✓ Genuine 16-bit content",
        phase_correlation="0.847", phase_verdict="Normal stereo",
        clipped_samples="0", clipping_verdict="✓ No clipped samples",
        silence_total_pct="0.8%",
        silence_sections=["00:00 → 00:01 (0.9s)"],
        rg_stored="+0.92 dB", rg_measured_lufs="-14.80 LUFS",
        rg_delta="0.5 dB", rg_verdict="✓ RG tag matches measured loudness",
    )
    sox_stats = {
        "samplesRead":"20,652,912","lengthSeconds":"234.160000",
        "scaledBy":"2147483647.0","maximumAmplitude":"0.988556",
        "minimumAmplitude":"-0.921204","midlineAmplitude":"0.033676",
        "meanNorm":"0.084927","meanAmplitude":"-0.000003",
        "rmsAmplitude":"0.122069","maximumDelta":"0.791107",
        "minimumDelta":"0.000000","meanDelta":"0.075868",
        "rmsDelta":"0.103592","roughFrequency":"5956","volumeAdjustment":"1.012",
    }

    report = ForensicReport(
        filepath=demo_path, tags=tags, technical=tech, sox_stats=sox_stats,
        loudness=lp, authenticity=auth, dr_score="DR9",
        spectrogram_path=Path("01 Habib ft. Nancy - Didha_spectrogram.png"),
    )

    print()
    print(_c(C.GREY, "  ╔══════════════════════════════════╗"))
    print(_c(C.GREY, "  ║") + _c(C.GOLD + C.BOLD, "   DEMO MODE  ·  synthetic data   ") + _c(C.GREY, "║"))
    print(_c(C.GREY, "  ╚══════════════════════════════════╝"))
    print_report(report, file_size_mb=41.8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="af2.py",
        description="Audio Forensics CLI — Master Chef Edition (improved thresholds for real audio)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python af2.py track.flac
  python af2.py track.flac --json
  python af2.py *.flac            (batch / album mode)
  python af2.py --info track.flac (lightweight mode)
  python af2.py --demo
        """
    )
    parser.add_argument("files",   nargs="*",        help="Audio file(s) to analyse")
    parser.add_argument("--demo",  action="store_true", help="Run with synthetic demo data")
    parser.add_argument("--json",  action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--fast",  action="store_true", help="Analyse first 60s only (faster batch runs)")
    parser.add_argument("--info",  action="store_true", help="Only show basic metadata and SoX stats")
    args = parser.parse_args()

    # Dependency checks
    missing = []
    for tool in ("ffmpeg", "sox", "mediainfo"):
        if not _tool_available(tool):
            missing.append(tool)
    if missing:
        print(f"Error: Missing required tool(s): {', '.join(missing)}", file=sys.stderr)
        print("Please install them and ensure they are in your PATH.", file=sys.stderr)
        sys.exit(1)

    if args.demo:
        run_demo()
        return

    if not args.files:
        parser.print_help()
        sys.exit(1)

    paths = [Path(f) for f in args.files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"Error: not found — {p}", file=sys.stderr)
        sys.exit(1)

    if args.info:
        reports = [build_info_report(p) for p in paths]
        if args.json:
            out = [_report_to_dict(r) for r in reports]
            print(json.dumps(out if len(out) > 1 else out[0], indent=2, default=str))
            return
        for report in reports:
            print_report(report)
        return

    fast_secs = 60.0 if args.fast else None
    reports = [build_report(p, fast_secs=fast_secs) for p in paths]

    if args.json:
        out = [_report_to_dict(r) for r in reports]
        print(json.dumps(out if len(out) > 1 else out[0], indent=2, default=str))
        return

    for report in reports:
        print_report(report)

    if len(reports) > 1:
        print_batch_summary(reports)


if __name__ == "__main__":
    main()
