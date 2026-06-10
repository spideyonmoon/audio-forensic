#!/usr/bin/env python3
"""
Audio Forensics CLI — comprehensive audio authenticity analysis
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
# ANSI palette & Helpers
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

def _c(colour: str, text: str) -> str: return f"{colour}{text}{C.RESET}"
def _kv(key: str, value: str, *, width: int = 26) -> str: return f"  {_c(C.CYAN, key.ljust(width))} {_c(C.WHITE, value)}" if value else ""
def _rule(char: str = "─", width: int = 62) -> str: return _c(C.GREY, char * width)
def _section(title: str) -> str: pad = max(0, 58 - len(title)); return f"\n{_c(C.GREY, '── ')}{_c(C.GOLD + C.BOLD, title)}{_c(C.GREY, ' ' + '─' * pad)}"
def _subsection(title: str) -> str: return f"\n  {_c(C.GREY, title)}"
def _camel_case(text: str) -> str:
    words = re.sub(r"[^a-zA-Z0-9 ]", "", text).split()
    return words[0].lower() + "".join(w.capitalize() for w in words[1:]) if words else ""

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)

def _tool_available(name: str) -> bool:
    checker = "where" if sys.platform == "win32" else "which"
    try: return subprocess.run([checker, name], capture_output=True, check=False).returncode == 0
    except FileNotFoundError: return False

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class AudioTags:
    title: str = ""; album: str = ""; date: str = ""; album_artist: str = ""
    artist: str = ""; bpm: str = ""; comment_quality: str = ""; comments: str = ""
    replaygain_track_gain: str = ""; replaygain_album_gain: str = ""

@dataclass
class AudioTechnical:
    bit_rate: str = ""; channels: str = ""; precision: str = ""; sample_rate: str = ""
    sample_encoding: str = ""; duration: str = ""; duration_sec: float = 0.0

@dataclass
class LoudnessProfile:
    peak_db: str = ""; rms_db: str = ""; rms_peak_db: str = ""; rms_trough_db: str = ""
    noise_floor_db: str = ""; dynamic_range_db: str = ""; crest_factor_db: str = ""
    flat_factor: str = ""; peak_count: str = ""; sox_entropy: str = ""; dc_offset: str = ""
    zero_crossings_rate: str = ""; lufs_integrated: str = ""; lufs_range: str = ""
    true_peak_dbtp: str = ""; lufs_momentary_max: str = ""; lufs_shortterm_max: str = ""
    apple_music_delta: str = ""; spotify_delta: str = ""

@dataclass
class SpectralAnalysis:
    cutoff_hz: float = 0.0; cutoff_hz_str: str = ""
    cutoff_variance: float = 0.0; cutoff_variance_interp: str = ""
    cutoff_sharpness_db: float = 0.0; cutoff_sharpness_interp: str = ""
    hf_energy_ratio: float = 0.0; hf_energy_interp: str = ""
    banding_score: float = 0.0; banding_interp: str = ""
    nf_above_cutoff_db: float = 0.0; nf_interp: str = ""
    side_anomaly_score: float = 0.0; side_interp: str = ""
    entropy: float = 0.0; entropy_interp: str = ""
    lpf_detected: bool = False; lpf_cutoff_str: str = ""
    dsd_detected: bool = False; lossy_score: int = 0
    natural_score: int = 0; net_score: int = 0; max_score: int = 0
    raw_lossy_pct: float = 0.0; net_confidence_pct: float = 0.0
    verdict_label: str = ""; primary_verdict: str = ""
    evidence: list[str] = field(default_factory=list)
    natural_evidence: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

@dataclass
class AuthenticityReport:
    spectral: "SpectralAnalysis | None" = None
    spectral_cutoff_hz: str = ""; spectral_cutoff_verdict: str = ""; lpf_detected: bool = False
    lpf_cutoff_hz: str = ""; bit_depth_authentic: str = ""; phase_correlation: str = ""
    phase_verdict: str = ""; clipped_samples: str = ""; clipping_verdict: str = ""
    silence_total_pct: str = ""; silence_sections: list[str] = field(default_factory=list)
    rg_stored: str = ""; rg_measured_lufs: str = ""; rg_delta: str = ""; rg_verdict: str = ""

@dataclass
class ForensicReport:
    filepath: Path; tags: AudioTags = field(default_factory=AudioTags)
    technical: AudioTechnical = field(default_factory=AudioTechnical)
    sox_stats: dict[str, str] = field(default_factory=dict)
    loudness: LoudnessProfile = field(default_factory=LoudnessProfile)
    authenticity: AuthenticityReport = field(default_factory=AuthenticityReport)
    dr_score: str = "N/A"; spectrogram_path: Optional[Path] = None
    @property
    def file_size_mb(self) -> float: return self.filepath.stat().st_size / (1024 * 1024)

# ---------------------------------------------------------------------------
# Tool Extractors
# ---------------------------------------------------------------------------
def extract_mediainfo(filepath: Path) -> tuple[AudioTags, AudioTechnical]:
    result = _run(["mediainfo", "--Output=JSON", str(filepath)])
    if result.returncode != 0: return AudioTags(), AudioTechnical()

    data = json.loads(result.stdout)
    tags, tech = AudioTags(), AudioTechnical()

    for track in data.get("media", {}).get("track", []):
        t = track.get("@type")
        if t == "General":
            extra = track.get("extra", {})
            tags.title = track.get("Title", "")
            tags.album = track.get("Album", "")
            tags.date = track.get("Recorded_Date", "")
            tags.album_artist = track.get("Album_Artist", "")
            tags.artist = track.get("Performer", "")
            tags.bpm = track.get("BPM", "")
            tags.comments = track.get("Comment", extra.get("Comment", ""))
            tags.comment_quality = extra.get("commentQuality", "")
            tags.replaygain_track_gain = extra.get("REPLAYGAIN_TRACK_GAIN", track.get("REPLAYGAIN_TRACK_GAIN", ""))
            tags.replaygain_album_gain = extra.get("REPLAYGAIN_ALBUM_GAIN", track.get("REPLAYGAIN_ALBUM_GAIN", ""))
        elif t == "Audio":
            bit_depth = track.get("BitDepth", "")
            fmt = track.get("Format", "").upper()
            if fmt == "MPEG AUDIO": fmt = "MP3"

            raw_br = track.get("BitRate")
            raw_dur = float(track.get("Duration", 0))
            tech.duration_sec = raw_dur
            tech.bit_rate = f"{int(raw_br) // 1000:,} kbps" if raw_br else ""
            tech.channels = track.get("Channels", "")
            tech.precision = f"{bit_depth}-bit" if bit_depth else ""
            tech.sample_rate = track.get("SamplingRate", "")
            tech.sample_encoding = f"{bit_depth}-bit {fmt}" if bit_depth else fmt
            mins, secs = divmod(int(raw_dur), 60)
            tech.duration = f"{mins:02d}:{secs:02d}"

    return tags, tech

_SOX_UNSUPPORTED = {".m4a", ".mp4", ".aac", ".ogg", ".opus", ".wma", ".ape", ".mp3"}

class _TempWAV:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._tmp: Optional[Path] = None

    def __enter__(self) -> Path:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self._tmp = Path(tmp)
        _run(["ffmpeg", "-y", "-i", str(self.filepath), "-vn", "-ac", "2", "-sample_fmt", "s16", str(self._tmp)])
        return self._tmp

    def __exit__(self, *_):
        if self._tmp and self._tmp.exists(): self._tmp.unlink(missing_ok=True)

def extract_sox_stats(filepath: Path) -> dict[str, str]:
    if filepath.suffix.lower() in _SOX_UNSUPPORTED:
        with _TempWAV(filepath) as wav:
            result = _run(["sox", str(wav), "-n", "stat"])
    else:
        result = _run(["sox", str(filepath), "-n", "stat"])

    stats: dict[str, str] = {}
    for line in result.stderr.splitlines():
        if ":" not in line: continue
        raw_key, _, raw_val = line.partition(":")
        if key := _camel_case(raw_key.strip()): stats[key] = raw_val.strip()
    return stats

def extract_loudness(filepath: Path) -> LoudnessProfile:
    r = _run(["ffmpeg", "-i", str(filepath), "-vn", "-af", "astats", "-f", "null", "-"])
    def _last(pattern: str) -> str:
        hits = re.findall(pattern, r.stderr)
        if not hits: return ""
        v = hits[-1]
        if v in ("inf", "-inf", "nan") or v.lower().startswith("n"): return ""
        try: return f"{float(v):.2f}"
        except ValueError: return v

    lp = LoudnessProfile()
    lp.peak_db = _last(r"Peak level dB:\s*([-\w.]+)")
    lp.rms_db = _last(r"RMS level dB:\s*([-\w.]+)")
    lp.rms_peak_db = _last(r"RMS peak dB:\s*([-\w.]+)")
    lp.rms_trough_db = _last(r"RMS trough dB:\s*([-\w.]+)")
    lp.noise_floor_db = _last(r"Noise floor dB:\s*([-\w.]+)")
    lp.dynamic_range_db = _last(r"Dynamic range:\s*([-\w.]+)")
    lp.crest_factor_db = _last(r"Crest factor:\s*([-\w.]+)")
    lp.flat_factor = _last(r"Flat factor:\s*([-\w.]+)")
    lp.peak_count = _last(r"Peak count:\s*([-\w.]+)")
    lp.sox_entropy = _last(r"Entropy:\s*([-\w.]+)")
    lp.dc_offset = _last(r"DC offset:\s*([-\w.]+)")
    lp.zero_crossings_rate = _last(r"Zero crossings rate:\s*([-\w.]+)")

    r2 = _run(["ffmpeg", "-i", str(filepath), "-vn", "-af", "aresample=48000,ebur128=peak=true", "-f", "null", "-"])
    def _field(pat: str) -> str:
        matches = re.findall(pat, r2.stderr)
        return matches[-1].strip() if matches else ""

    lp.lufs_integrated = _field(r"I:\s*([-\d.]+)\s*LUFS")
    lp.lufs_range = _field(r"LRA:\s*([\d.]+)\s*LU")
    lp.true_peak_dbtp = _field(r"True peak:\s*([-\d.]+)\s*dBTP")
    lp.lufs_momentary_max = _field(r"Momentary max:\s*([-\d.]+)\s*LUFS")
    lp.lufs_shortterm_max = _field(r"Short-term max:\s*([-\d.]+)\s*LUFS")

    if lp.lufs_integrated:
        try:
            measured = float(lp.lufs_integrated)
            lp.apple_music_delta = f"{-16.0 - measured:+.1f} dB"
            lp.spotify_delta = f"{-14.0 - measured:+.1f} dB"
        except ValueError: pass

    return lp

def measure_dynamic_range(filepath: Path) -> str:
    result = _run(["ffmpeg", "-i", str(filepath), "-vn", "-af", "drmeter", "-f", "null", "-"])
    match = re.search(r"DR:\s+([\d.]+)", result.stderr)
    return f"DR{int(float(match.group(1)))}" if match else "N/A"

def check_bit_depth_authenticity(filepath: Path, claimed_depth: int) -> str:
    if not claimed_depth: return ""

    if claimed_depth == 24 and _NUMPY_OK:
        try:
            cmd = ["ffmpeg", "-i", str(filepath), "-vn", "-t", "5", "-f", "s24le", "-acodec", "pcm_s24le", "pipe:1"]
            result = subprocess.run(cmd, capture_output=True, check=False)

            raw_bytes = result.stdout
            if raw_bytes:
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                lsb_array = arr[0::3]

                if np.any(lsb_array != 0):
                    return "✓ Genuine 24-bit content [Numpy Binary Scan]"
                else:
                    return "⚠ 16-bit content padded into 24-bit container [Numpy Binary Scan]"
        except Exception:
            pass

    return f"✓ Genuine {claimed_depth}-bit content"

def measure_phase_correlation(filepath: Path, channels: int) -> tuple[str, str]:
    if channels < 2: return "", ""
    r = _run(["ffmpeg", "-i", str(filepath), "-vn", "-af", "aphasemeter=r=10", "-f", "null", "-"])
    vals = re.findall(r"phase=([-\d.]+)", r.stderr)
    if not vals: return "", ""
    try:
        avg = sum(float(v) for v in vals) / len(vals)
        if avg >= 0.9: return f"{avg:.3f}", "Mono-compatible"
        elif avg >= 0.5: return f"{avg:.3f}", "Normal stereo"
        elif avg >= 0.0: return f"{avg:.3f}", "Wide stereo"
        elif avg >= -0.3: return f"{avg:.3f}", "⚠ Possible fake stereo / heavy M-S processing"
        else: return f"{avg:.3f}", "⚠ Phase cancellation — check mono fold-down"
    except ValueError: return "", ""

def detect_clipping(filepath: Path) -> tuple[str, str]:
    r = _run(["ffmpeg", "-i", str(filepath), "-vn", "-af", "astats=clipping=1", "-f", "null", "-"])
    counts = re.findall(r"Number of clippings:\s*(\d+)", r.stderr)
    if not counts: return "", ""
    total = sum(int(c) for c in counts)
    if total == 0: return "0", "✓ No clipped samples"
    elif total < 10: return str(total), f"~ {total} clipped sample(s) — minor"
    else: return str(total), f"⚠ {total:,} clipped samples — audible distortion likely"

def map_silence(filepath: Path, duration_sec: float) -> tuple[str, list[str]]:
    r = _run(["ffmpeg", "-i", str(filepath), "-vn", "-af", "silencedetect=noise=-60dB:d=0.5", "-f", "null", "-"])
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", r.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", r.stderr)]
    eof_padded = False
    if len(starts) > len(ends):
        ends.append(duration_sec)
        eof_padded = True
    total_silent = sum(e - s for s, e in zip(starts, ends))
    pct = (total_silent / duration_sec * 100) if duration_sec > 0 else 0
    sections = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        marker = " → EOF" if eof_padded and i == len(starts) - 1 else ""
        sections.append(f"{int(s//60):02d}:{int(s%60):02d} → {int(e//60):02d}:{int(e%60):02d} ({e-s:.1f}s){marker}")
    return f"{pct:.1f}%", sections

def audit_replaygain(tags: AudioTags, lufs_integrated: str) -> tuple[str, str, str, str]:
    stored_raw = tags.replaygain_track_gain.strip()
    if not stored_raw or not lufs_integrated: return stored_raw, lufs_integrated, "", ""
    try:
        stored_db = float(re.sub(r"[^\d.\-]", "", stored_raw.split()[0]))
        measured_lufs = float(lufs_integrated)
        implied_level = -18.0 + stored_db
        delta = abs(implied_level - measured_lufs)
        if delta < 1.0: verdict = "✓ RG tag matches measured loudness"
        elif delta < 3.0: verdict = f"~ {delta:.1f} dB mismatch — minor discrepancy"
        else: verdict = f"⚠ {delta:.1f} dB mismatch — file may have been re-encoded after tagging"
        return stored_raw, f"{measured_lufs:.2f} LUFS", f"{delta:.1f} dB", verdict
    except (ValueError, IndexError): return stored_raw, lufs_integrated, "", ""


def generate_spectrogram(filepath: Path) -> Path:
    """Generates a clean mono spectrogram.

    Strategy:
    1. Mix audio to mono WAV via FFmpeg (universal decode).
    2. Generate spectrogram from mono WAV using SoX (best visual quality).
    3. If SoX fails for any reason, fall back to FFmpeg showspectrumpic.
    """
    output = filepath.with_name(f"{filepath.stem}_spectrogram.png")
    tmp_mono: Optional[Path] = None

    try:
        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="af_spec_")
        os.close(fd)
        tmp_mono = Path(tmp)

        decode_result = _run([
            "ffmpeg", "-y", "-i", str(filepath), "-vn",
            "-ac", "1",           # mix to mono
            "-sample_fmt", "s16", # 16-bit PCM
            str(tmp_mono)
        ])

        if decode_result.returncode == 0 and tmp_mono.exists() and tmp_mono.stat().st_size > 0:
            sox_result = _run([
                "sox", str(tmp_mono), "-n",
                "spectrogram",
                "-x", "1280",   # width in pixels
                "-y", "512",    # height in pixels
                "-z", "120",    # dynamic range in dB
                "-Z", "-20",    # clip ceiling at −20 dB (removes whitewash)
                "-t", filepath.stem,
                "-o", str(output)
            ])
            if sox_result.returncode == 0 and output.exists():
                return output

        _run([
            "ffmpeg", "-y", "-i", str(filepath), "-vn",
            "-lavfi", "showspectrumpic=s=1280x512:mode=combined:color=fiery:legend=1",
            str(output)
        ])
    finally:
        if tmp_mono and tmp_mono.exists():
            tmp_mono.unlink(missing_ok=True)

    return output

# ---------------------------------------------------------------------------
# SpectralEngine — numpy FFT-based authenticity analysis
# ---------------------------------------------------------------------------
try: import numpy as np; _NUMPY_OK = True
except ImportError: _NUMPY_OK = False

class SpectralEngine:
    WINDOW = 4096; HOP = 2048; CUTOFF_DB = -65.0; NYQUIST_MARGIN = 0.85
    SCORE_CUTOFF_WELL_BELOW_NYQUIST = 2; SCORE_SHARP_CLIFF_HARD = 3; SCORE_SHARP_CLIFF_SOFT = 1
    SCORE_HF_NEAR_ZERO = 1; SCORE_VOID_ABOVE_CUTOFF = 3; SCORE_QUIET_ABOVE_CUTOFF = 1
    SCORE_VERY_STABLE_CUTOFF = 1; SCORE_BANDING_STRONG = 1; SCORE_SIDE_ANOMALY = 2
    MAX_LOSSY_SCORE = 14
    NATURAL_GRADUAL_ROLLOFF = 1; NATURAL_HIGH_VARIANCE = 1; NATURAL_MODERATE_VARIANCE = 1
    NATURAL_RICH_HF = 1; NATURAL_HF_NOISE = 1; NATURAL_HEALTHY_SIDE = 1; NATURAL_HIGH_ENTROPY = 1
    MP3_CUTOFFS = {320: 20500, 256: 20000, 192: 19000, 160: 18500, 128: 16000, 96: 15500, 64: 12000}

    def __init__(self, filepath: Path, sample_rate: int):
        self.filepath = filepath; self.sample_rate = sample_rate; self.nyquist = sample_rate / 2.0

    def _decode_audio(self, max_seconds: Optional[float] = None) -> "np.ndarray | None":
        if not _NUMPY_OK: return None
        cmd = ["ffmpeg", "-i", str(self.filepath), "-vn"]
        if max_seconds: cmd += ["-t", str(max_seconds)]
        cmd += ["-ac", "1", "-ar", str(self.sample_rate), "-f", "f32le", "pipe:1"]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0 or not result.stdout: return None
        return np.frombuffer(result.stdout, dtype=np.float32)

    def _decode_stereo(self, max_seconds: Optional[float] = None) -> "tuple[np.ndarray, np.ndarray] | None":
        if not _NUMPY_OK: return None
        cmd = ["ffmpeg", "-i", str(self.filepath), "-vn"]
        if max_seconds: cmd += ["-t", str(max_seconds)]
        cmd += ["-ac", "2", "-ar", str(self.sample_rate), "-f", "f32le", "pipe:1"]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0 or not result.stdout: return None
        raw = np.frombuffer(result.stdout, dtype=np.float32)
        if len(raw) < 2: return None
        interleaved = raw.reshape(-1, 2)
        return (interleaved[:, 0] + interleaved[:, 1]) / 2.0, (interleaved[:, 0] - interleaved[:, 1]) / 2.0

    def _compute_frames(self, audio: "np.ndarray") -> "np.ndarray":
        win = np.hanning(self.WINDOW); frames = []
        for i in range(0, len(audio) - self.WINDOW, self.HOP):
            frames.append(np.abs(np.fft.rfft(audio[i:i + self.WINDOW] * win)))
        return np.array(frames)

    def _freq_bins(self) -> "np.ndarray": return np.fft.rfftfreq(self.WINDOW, 1.0 / self.sample_rate)

    @staticmethod
    def _interp_variance(var: float, legit_cutoff: bool) -> str:
        if var < 1000: return "[rigid/encoded-like]"
        elif var < 10000: return "[stable: normal for mastered audio]"
        elif var < 100000: return "[moderate: natural organic fluctuation]"
        elif var < 1000000: return "[high variation: organic/analog source]" if legit_cutoff else "[erratic cutoff: typical of VBR lossy encoders like AAC/Opus]"
        else: return "[very high variation: complex analog source]" if legit_cutoff else "[erratic cutoff: typical of VBR lossy encoders like AAC/Opus]"

    @staticmethod
    def _interp_sharpness(s: float) -> str:
        if s < 2: return "[gradual: natural EQ / mastering]"
        elif s < 5: return "[moderate: normal variation]"
        elif s < 15: return "[steep: algorithmic filter possible]"
        else: return "[sharp cliff: hard mathematical low-pass filter]"

    @staticmethod
    def _interp_hf_ratio(r: float) -> str:
        if r < 0.005: return "[energy depletion: possible aggressive filter]"
        elif r < 0.015: return "[low: typical for mastered/pop audio]"
        elif r < 0.05: return "[moderate: normal mastered audio]"
        else: return "[rich: full-spectrum, dynamic recording]"

    @staticmethod
    def _interp_banding(b: float) -> str:
        if b < 0.7: return "[minimal: no heavy quantization artifacts]"
        elif b < 0.85: return "[moderate: normal for 16-bit PCM]"
        elif b < 0.95: return "[strong: expected in PCM sources]"
        else: return "[severe: heavy quantization detected]"

    @staticmethod
    def _interp_nf(nf: float) -> str:
        if nf < -80: return "[silent void: suspicious digital cutoff]"
        elif nf < -55: return "[very quiet: typical digital silence]"
        elif nf < -35: return "[moderate: natural dither or tape hiss]"
        else: return "[loud: heavy analog noise or DSD shaping]"

    @staticmethod
    def _interp_side(a: float) -> str:
        if a < 0.15: return "[healthy: wide, complex stereo]"
        elif a < 0.30: return "[normal: typical stereo imaging]"
        elif a < 0.50: return "[mild depletion: acceptable joint stereo]"
        elif a < 0.70: return "[moderate anomaly: heavy joint stereo]"
        else: return "[severe anomaly: artificial stereo width or heavy compression]"

    @staticmethod
    def _interp_entropy(e: float, legit_cutoff: bool) -> str:
        if e < 7.0: return "[low: simple/tonal content]"
        elif e < 8.5: return "[moderate: typical music complexity]"
        elif e < 9.5: return "[high: complex/dynamic content]" if legit_cutoff else "[high entropy: lossy noise-shaping / VBR footprint]"
        else: return "[very high: noise-like complexity]" if legit_cutoff else "[very high entropy: lossy ultrasonic noise / dithering]"

    def _cutoff_per_frame(self, frames: "np.ndarray", bins: "np.ndarray") -> "np.ndarray":
        cutoffs = []
        for frame in frames:
            ref = frame.max() + 1e-12
            db = 20.0 * np.log10(frame / ref + 1e-12)
            above = np.where(db > self.CUTOFF_DB)[0]
            cutoffs.append(float(bins[above[-1]]) if len(above) else 0.0)
        return np.array(cutoffs)

    def _sharpness(self, frames: "np.ndarray", bins: "np.ndarray", cutoff_hz: float, window_hz: float = 2500.0) -> float:
        bin_hz = bins[1] - bins[0]
        lo = max(0, int((cutoff_hz - window_hz) / bin_hz))
        hi = min(len(bins), int((cutoff_hz + window_hz * 0.25) / bin_hz))
        avg = frames.mean(axis=0)
        db = 20.0 * np.log10(avg[lo:hi] / (avg.max() + 1e-12) + 1e-12)
        return float(np.abs(np.diff(db)).max()) if len(db) > 1 else 0.0

    def _hf_energy_ratio(self, frames: "np.ndarray", bins: "np.ndarray", threshold_hz: float = 15000.0) -> float:
        return float(frames[:, int(threshold_hz / (bins[1] - bins[0])):].sum()) / (float(frames.sum()) + 1e-12)

    def _banding_score(self, frames: "np.ndarray", bins: "np.ndarray", cutoff_hz: float, scan_hz: float = 1500.0) -> float:
        bin_hz = bins[1] - bins[0]
        hi = int(cutoff_hz / bin_hz)
        lo = max(0, hi - int(scan_hz / bin_hz))
        region = frames[:, lo:hi]
        if region.shape[1] < 4: return 0.0

        ref = region.max() + 1e-12
        db = 20.0 * np.log10(region / ref + 1e-12)

        temporal_std = np.std(db, axis=0).mean()
        return float(np.clip(1.0 - (temporal_std / 15.0), 0.0, 1.0))

    def _noise_floor_above_cutoff(self, frames: "np.ndarray", bins: "np.ndarray", cutoff_hz: float) -> float:
        above = frames[:, int(cutoff_hz / (bins[1] - bins[0])):]
        if above.size == 0: return -120.0
        return float(20.0 * np.log10(float(np.sqrt(np.mean(above ** 2))) + 1e-12))

    def _side_channel_anomaly(self, mid: "np.ndarray", side: "np.ndarray", bins: "np.ndarray") -> float:
        if not _NUMPY_OK or mid is None or side is None or len(mid) < self.WINDOW * 2: return 0.0
        mid_frames = self._compute_frames(mid)
        side_frames = self._compute_frames(side)

        bin_hz = bins[1] - bins[0]
        idx_10k = int(10000 / bin_hz)
        if idx_10k >= mid_frames.shape[1]: return 0.0

        mid_hf = mid_frames[:, idx_10k:]
        side_hf = side_frames[:, idx_10k:]

        e_ratio = float(np.mean(side_hf)) / (float(np.mean(mid_hf)) + 1e-12)
        score, wt = 0.0, 1.0
        if e_ratio < 0.02: score += 1.0
        elif e_ratio < 0.08: score += 0.6
        return float(score / wt)

    def _lpf_scan(self, frames: "np.ndarray", bins: "np.ndarray") -> tuple[bool, str]:
        thz = self.nyquist * 0.90
        bin_hz = bins[1] - bins[0]
        top_idx = int(thz / bin_hz)
        top = frames[:, top_idx:]
        if top.size == 0: return False, ""
        if (float(top.sum()) / (float(frames.sum()) + 1e-12)) >= 0.00005: return False, ""

        avg = frames.mean(axis=0)
        ref = avg.max() + 1e-12
        for i in range(top_idx, 0, -1):
            if 20 * np.log10(avg[i] / ref + 1e-12) > -40.0:
                return True, f"~{int(bins[i] / 1000)}kHz"
        return True, "< 1kHz"

    def _dsd_scan(self, frames: "np.ndarray", bins: "np.ndarray") -> bool:
        if self.sample_rate <= 48000: return False
        bin_hz = bins[1] - bins[0]
        idx_20k = int(20000 / bin_hz)
        idx_30k = int(30000 / bin_hz)
        if idx_30k >= frames.shape[1]: return False
        avg = frames.mean(axis=0)
        return bool((avg[idx_30k:].mean() + 1e-12) > (avg[int(15000/bin_hz):idx_20k].mean() + 1e-12) * 1.5)

    def _spectral_entropy(self, frames: "np.ndarray") -> float:
        avg = frames.mean(axis=0)
        p = (avg / (avg.sum() + 1e-12)); p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def _score(self, cutoff_hz: float, variance: float, sharpness: float, hf_ratio: float, nf_above: float, banding: float, side_anomaly: float, entropy: float, dsd_detected: bool) -> tuple[int, list[str], int, list[str]]:
        l_score, l_ev, n_score, n_ev = 0, [], 0, []

        if cutoff_hz < self.nyquist * 0.85 and cutoff_hz < 18500:
            l_score += self.SCORE_CUTOFF_WELL_BELOW_NYQUIST
            l_ev.append(f"Premature High-Frequency Rolloff: Hard cutoff detected at {cutoff_hz:,.0f} Hz, strongly suggesting lossy perceptual encoding.")
        if sharpness > 15.0:
            l_score += self.SCORE_SHARP_CLIFF_HARD
            l_ev.append(f"Unnatural Spectral Cliff: Frequency drop-off is mathematically steep ({sharpness:.1f} dB/bin), typical of algorithmic low-pass filters.")
        elif sharpness > 8.0:
            l_score += self.SCORE_SHARP_CLIFF_SOFT
            l_ev.append(f"Steep Frequency Ceiling: Substantial spectral cliff detected ({sharpness:.1f} dB/bin).")
        if hf_ratio < 0.005:
            l_score += self.SCORE_HF_NEAR_ZERO
            l_ev.append(f"Energy Depletion: Insufficient high-frequency energy ratio ({hf_ratio:.4f}), often caused by acoustic masking compression.")
        if nf_above < -70.0:
            l_score += self.SCORE_VOID_ABOVE_CUTOFF
            l_ev.append(f"Digital Void: Lack of natural noise floor above the cutoff threshold ({nf_above:.1f} dB) indicates discarded data rather than analog warmth.")
        elif nf_above < -40.0:
            l_score += self.SCORE_QUIET_ABOVE_CUTOFF
            l_ev.append(f"Attenuated Noise Floor: Unusually quiet spectrum above the primary frequency ceiling ({nf_above:.1f} dB).")
        if variance < 1000.0 and cutoff_hz < self.nyquist * 0.85:
            l_score += self.SCORE_VERY_STABLE_CUTOFF
            l_ev.append(f"Rigid Cutoff Variance: Frequency ceiling lacks natural fluctuation ({variance:.1f} Hz²), pointing to a hard-coded digital filter.")
        if banding > 0.92 and cutoff_hz < self.nyquist * 0.80:
            l_score += self.SCORE_BANDING_STRONG
            l_ev.append(f"Quantization Artifacts: Strong frequency banding detected ({banding:.2f}).")
        if side_anomaly > 0.60:
            l_score += self.SCORE_SIDE_ANOMALY
            l_ev.append(f"Stereo Degradation: Extreme side-channel anomaly detected ({side_anomaly:.2f}), suggesting destructive joint-stereo compression.")

        if dsd_detected:
            l_ev.append("Ultrasonic Noise Shaping: Massive high-frequency energy slope detected, highly indicative of a DSD/SACD transcode.")
        else:
            if hf_ratio > 0.05 and cutoff_hz > self.nyquist * 0.85:
                n_score += self.NATURAL_RICH_HF
                n_ev.append(f"Rich Harmonic Extension: Abundant high-frequency energy consistent with lossless preservation.")
            if nf_above > -50.0:
                n_score += self.NATURAL_HF_NOISE
                n_ev.append(f"Preserved Noise Floor: Presence of natural dither or analog hiss above the primary frequency ceiling.")
            if entropy > 8.5 and cutoff_hz > self.nyquist * 0.85:
                n_score += self.NATURAL_HIGH_ENTROPY
                n_ev.append(f"Spectral Complexity: High entropy score indicates dense, unpredictable signal data devoid of aggressive compression.")

        if sharpness < 5.0:
            n_score += self.NATURAL_GRADUAL_ROLLOFF
            n_ev.append(f"Organic Frequency Rolloff: Gradual attenuation consistent with natural acoustic decay or analog mastering.")
        if variance > 100000 and not dsd_detected and cutoff_hz > self.nyquist * 0.85:
            n_score += self.NATURAL_HIGH_VARIANCE
            n_ev.append(f"Dynamic Cutoff Variance: Frequency ceiling fluctuates organically, typical of uncompressed analog-to-digital transfers.")
        elif variance > 10000 and cutoff_hz > self.nyquist * 0.85:
            n_score += self.NATURAL_MODERATE_VARIANCE
            n_ev.append(f"Healthy Cutoff Variance: Frequency ceiling exhibits natural, subtle fluctuations.")
        if side_anomaly < 0.2:
            n_score += self.NATURAL_HEALTHY_SIDE
            n_ev.append(f"Phase & Stereo Integrity: Wide, complex side-channel information preserved without joint-stereo artifacts.")

        return l_score, l_ev, n_score, n_ev

    def _verdict(self, net_score: int, cutoff_hz: float, dsd_detected: bool) -> tuple[str, str, list[str]]:
        caveats = [
            "Analog Origins: Vinyl and tape transfers naturally exhibit HF rolloff and higher noise floors; these are not suspicious traits.",
            "Modern Mastering: Audio engineers frequently apply gentle low-pass filters at 19-20 kHz to prevent aliasing distortion.",
            "Transcode Artifacts: Lossless encoders (FLAC/ALAC) will perfectly preserve lossy characteristics if the source material was already degraded prior to encoding."
        ]
        ext = self.filepath.suffix.lower()
        if ext in {".mp3", ".aac", ".ogg", ".opus", ".wma"}:
            mp3_match = ""
            for br, freq in sorted(self.MP3_CUTOFFS.items(), reverse=True):
                if abs(cutoff_hz - freq) <= 300 and cutoff_hz < 20000: mp3_match = f" — matches ~{br}kbps MP3 encoder profile"; break
            sentence = f"ℹ Natively Lossy Format ({ext.upper()}){mp3_match}"
            if not mp3_match and net_score >= 6: sentence += " — severe degradation detected."
            return "CAUTION", sentence, []
        if dsd_detected: caveats.append("DSD transcode detected. Ultrasonic noise inflates entropy and HF scores.")
        if net_score >= 6: return "SUSPICIOUS", "⚠  Spectral anomalies detected", caveats
        elif net_score >= 3: return "CAUTION", "~  Minor spectral quirks — likely legitimate", caveats
        elif net_score >= 1: return "LIKELY_GENUINE", "✓  Consistent with genuine lossless source", caveats
        else: return "GENUINE", "✓  Strong evidence of authentic lossless source", caveats

    def analyse(self, max_seconds: Optional[float] = None) -> SpectralAnalysis:
        result = SpectralAnalysis()
        if not _NUMPY_OK:
            result.primary_verdict = "numpy not installed"; result.verdict_label = "INCONCLUSIVE"; return result
        audio = self._decode_audio(max_seconds)
        if audio is None or len(audio) < self.WINDOW * 2:
            result.primary_verdict = "Could not decode audio"; result.verdict_label = "INCONCLUSIVE"; return result

        frames, bins = self._compute_frames(audio), self._freq_bins()
        if frames.shape[0] < 4:
            result.primary_verdict = "File too short"; result.verdict_label = "INCONCLUSIVE"; return result

        cutoffs_per_frame = self._cutoff_per_frame(frames, bins)
        cutoff_hz, cutoff_var = float(np.percentile(cutoffs_per_frame, 95)), float(np.var(cutoffs_per_frame))
        sharpness, hf_ratio = self._sharpness(frames, bins, cutoff_hz), self._hf_energy_ratio(frames, bins)
        banding, nf_above = self._banding_score(frames, bins, cutoff_hz), self._noise_floor_above_cutoff(frames, bins, cutoff_hz)
        lpf_detected, lpf_s = self._lpf_scan(frames, bins)
        entropy, dsd_detected = self._spectral_entropy(frames), self._dsd_scan(frames, bins)

        side_anomaly = 0.0
        if (stereo_pair := self._decode_stereo(max_seconds)) is not None:
            side_anomaly = self._side_channel_anomaly(stereo_pair[0], stereo_pair[1], bins)

        lossy_score, lossy_ev, natural_score, natural_ev = self._score(cutoff_hz, cutoff_var, sharpness, hf_ratio, nf_above, banding, side_anomaly, entropy, dsd_detected)
        net_score = max(0, lossy_score - natural_score)
        label, sentence, caveats = self._verdict(net_score, cutoff_hz, dsd_detected)

        legit_cutoff = cutoff_hz > (self.nyquist * 0.85)

        result.cutoff_hz, result.cutoff_hz_str = cutoff_hz, f"{int(cutoff_hz):,} Hz"
        result.cutoff_variance, result.cutoff_variance_interp = cutoff_var, self._interp_variance(cutoff_var, legit_cutoff)
        result.cutoff_sharpness_db, result.cutoff_sharpness_interp = sharpness, self._interp_sharpness(sharpness)
        result.hf_energy_ratio, result.hf_energy_interp = hf_ratio, self._interp_hf_ratio(hf_ratio)
        result.banding_score, result.banding_interp = banding, self._interp_banding(banding)
        result.nf_above_cutoff_db, result.nf_interp = nf_above, self._interp_nf(nf_above)
        result.side_anomaly_score, result.side_interp = side_anomaly, self._interp_side(side_anomaly)
        result.entropy, result.entropy_interp = entropy, self._interp_entropy(entropy, legit_cutoff)
        result.lpf_detected, result.lpf_cutoff_str, result.dsd_detected = lpf_detected, lpf_s, dsd_detected
        result.lossy_score, result.natural_score, result.net_score, result.max_score = lossy_score, natural_score, net_score, self.MAX_LOSSY_SCORE
        result.raw_lossy_pct = min(100.0, lossy_score / self.MAX_LOSSY_SCORE * 100.0) if lossy_score > 0 else 0.0
        result.net_confidence_pct = min(100.0, net_score / self.MAX_LOSSY_SCORE * 100.0) if net_score > 0 else 0.0
        result.verdict_label, result.primary_verdict = label, sentence
        result.evidence, result.natural_evidence, result.caveats = lossy_ev, natural_ev, caveats
        return result

# ---------------------------------------------------------------------------
# Report Building
# ---------------------------------------------------------------------------
def build_report(filepath: Path, fast_secs: Optional[float] = None) -> ForensicReport:
    tags, tech = extract_mediainfo(filepath)
    sox = extract_sox_stats(filepath)
    lp = extract_loudness(filepath)
    dr = measure_dynamic_range(filepath)
    spec_path = generate_spectrogram(filepath)

    try:
        sample_rate = int(tech.sample_rate.strip())
    except (ValueError, AttributeError):
        print("Warning: could not parse sample_rate, defaulting to 44100", file=sys.stderr)
        sample_rate = 44100

    try:
        channels = int(tech.channels.strip())
    except (ValueError, AttributeError):
        channels = 2
    try: claimed_depth = int(tech.precision.replace("-bit", "").strip())
    except ValueError: claimed_depth = 0

    auth = AuthenticityReport()
    engine = SpectralEngine(filepath, sample_rate)
    auth.spectral = engine.analyse(max_seconds=fast_secs)

    auth.spectral_cutoff_hz = auth.spectral.cutoff_hz_str
    auth.spectral_cutoff_verdict = auth.spectral.primary_verdict
    auth.lpf_detected = auth.spectral.lpf_detected
    auth.lpf_cutoff_hz = auth.spectral.lpf_cutoff_str
    auth.bit_depth_authentic = check_bit_depth_authenticity(filepath, claimed_depth)
    auth.phase_correlation, auth.phase_verdict = measure_phase_correlation(filepath, channels)
    auth.clipped_samples, auth.clipping_verdict = detect_clipping(filepath)
    auth.silence_total_pct, auth.silence_sections = map_silence(filepath, tech.duration_sec)
    auth.rg_stored, auth.rg_measured_lufs, auth.rg_delta, auth.rg_verdict = audit_replaygain(tags, lp.lufs_integrated)

    return ForensicReport(filepath=filepath, tags=tags, technical=tech, sox_stats=sox, loudness=lp, authenticity=auth, dr_score=dr, spectrogram_path=spec_path)

def build_info_report(filepath: Path) -> ForensicReport:
    tags, tech = extract_mediainfo(filepath)
    return ForensicReport(filepath=filepath, tags=tags, technical=tech, sox_stats=extract_sox_stats(filepath))

# ---------------------------------------------------------------------------
# Display helpers (Terminal output alignments)
# ---------------------------------------------------------------------------
def _fv(v: Optional[float]) -> Optional[float]:
    try: return float(v)
    except (TypeError, ValueError): return None

def _db_val(s: str) -> Optional[float]: return _fv(s)

def _dr_assessment(score: str) -> tuple[str, str]:
    try:
        n = int(score.replace("DR", ""))
        if n >= 14: return C.GREEN, "Highly dynamic (Audiophile / Classical / Vinyl)"
        elif n >= 10: return C.GREEN, "Excellent dynamic range (Standard mastered)"
        elif n >= 8: return C.WHITE, "Good dynamic range (Modern pop/rock standard)"
        elif n >= 5: return C.YELLOW, "Compressed (Loudness war casualty)"
        else: return C.RED, "Severely compressed (Brickwalled)"
    except ValueError: return C.WHITE, "Unknown"

def _peak_colour(db: str) -> str:
    v = _db_val(db)
    if v is None: return C.WHITE
    if v >= -0.1: return C.ORANGE
    if v >= -0.5: return C.YELLOW
    return C.GREEN

def _noise_colour(db: str) -> str:
    v = _db_val(db)
    if v is None: return C.WHITE
    if v <= -90: return C.GREEN
    if v <= -70: return C.YELLOW
    return C.RED

def _rms_colour(db: str) -> str:
    v = _db_val(db)
    if v is None: return C.WHITE
    if -18 <= v <= -10: return C.GREEN
    if v > -10: return C.RED
    return C.BLUE

def _lufs_colour(lufs: str) -> str:
    v = _db_val(lufs)
    if v is None: return C.WHITE
    if -16 <= v <= -12: return C.GREEN
    if v > -10: return C.RED
    return C.YELLOW

def _crest_colour(db: str) -> str:
    v = _db_val(db)
    if v is None: return C.WHITE
    if v >= 12: return C.GREEN
    if v >= 8: return C.WHITE
    if v >= 5: return C.YELLOW
    if v >= 3: return C.ORANGE
    return C.RED

def _flat_colour(v: str) -> str:
    try: return C.GREEN if float(v) == 0 else (C.YELLOW if float(v) <= 1 else C.RED)
    except ValueError: return C.WHITE

def _sox_entropy_colour(v: str) -> str:
    try:
        f = float(v)
        if f < 0.4: return C.GREEN
        elif f < 0.6: return C.WHITE
        else: return C.YELLOW
    except ValueError: return C.WHITE

def _sox_entropy_interp(v: str) -> str:
    try:
        f = float(v)
        if f < 0.1: return "[very low: highly tonal/structured]"
        elif f < 0.3: return "[low: typical music]"
        elif f < 0.5: return "[moderate: complex dynamics]"
        elif f < 0.7: return "[high: noisy or unusual content]"
        else: return "[very high: noise-like signal]"
    except ValueError: return ""

def _delta_colour(delta_str: str) -> str:
    try:
        v = float(delta_str.replace(" dB", "").replace("+", ""))
        if v > 0: return C.BLUE
        if v < -3: return C.RED
        return C.GREEN
    except ValueError: return C.WHITE

def _db(val: str, suffix: str = " dBFS") -> str:
    if val and not any(val.endswith(s) for s in ("dB", "dBFS", "LUFS", "dBTP", "LU")): return f"{val}{suffix}"
    return val

def _channel_label(raw: str) -> str: return {"1": "Mono", "2": "Stereo", "6": "5.1 Surround", "8": "7.1 Surround"}.get(raw.strip(), raw)
def _hz_label(raw: str) -> str:
    try: return f"{int(raw):,} Hz"
    except ValueError: return raw
def _fmt_stat_key(key: str) -> str: return re.sub(r"([A-Z])", r" \1", key).strip().title()

def _headroom_bar(noise_db: str, rms_db: str, peak_db: str, *, width: int = 42) -> list[str]:
    RANGE_MIN, RANGE_MAX = -120.0, 0.0
    span = RANGE_MAX - RANGE_MIN
    def _pct(s: str) -> Optional[float]:
        v = _db_val(s)
        return max(0.0, min(1.0, (v - RANGE_MIN) / span)) if v is not None else None
    nf, rm, pk = _pct(noise_db), _pct(rms_db), _pct(peak_db)
    if any(x is None for x in (nf, rm, pk)): return []
    bar = []
    for i in range(width):
        p = i / width
        if p < nf: bar.append(_c(C.GREY, "·"))
        elif p < rm: bar.append(_c(C.BLUE, "▒"))
        elif p < pk: bar.append(_c(C.GREEN, "█"))
        else: bar.append(_c(C.GREY, " "))
    pc = int(pk * width)
    if 0 <= pc < width: bar[pc] = _c(_peak_colour(peak_db), "▐")
    return [
        f"  {_c(C.GREY, '[')} {''.join(bar)} {_c(C.GREY, ']')}",
        f"   {_c(C.GREY, '-120' + ' ' * 12 + '-60' + ' ' * 9 + '-30' + ' ' * 5 + '-10  0 dBFS')}",
        f"   {_c(C.GREY,'·')} noise  {_c(C.BLUE,'▒')} RMS  {_c(C.GREEN,'█')} signal  {_c(_peak_colour(peak_db),'▐')} peak",
    ]

_CLIP_KEYS = {"maximumAmplitude", "minimumAmplitude"}
def _sox_amplitude_colour(key: str, raw: str) -> str:
    if key not in _CLIP_KEYS: return C.WHITE
    try:
        val = float(raw)
        if key == "maximumAmplitude": return C.RED if val > 1.0 else C.YELLOW if val >= 0.9999 else C.GREEN
        elif key == "minimumAmplitude": return C.RED if val < -1.0 else C.YELLOW if val <= -0.9999 else C.GREEN
    except ValueError: return C.WHITE
    return C.WHITE

def print_report(report: ForensicReport, *, file_size_mb: Optional[float] = None) -> None:
    t, tec, lp, auth, sz = report.tags, report.technical, report.loudness, report.authenticity, file_size_mb if file_size_mb is not None else report.file_size_mb
    W = 62; print(); print(_rule("═", W)); print(f"  {_c(C.BOLD + C.WHITE, report.filepath.name)}"); print(_rule("═", W))
    print(_section("IDENTITY"))
    for row in [_kv("Duration", tec.duration), _kv("BPM", t.bpm), _kv("File Size", f"{sz:.1f} MB")]:
        if row: print(row)
    print(_section("TAGS"))
    for row in [_kv("Title", t.title), _kv("Artist", t.artist), _kv("Album", t.album), _kv("Album Artist", t.album_artist), _kv("Year", t.date), _kv("Comment", t.comments), _kv("Rip Quality", t.comment_quality)]:
        if row: print(row)
    print(_section("TECHNICAL"))
    for row in [_kv("Encoding", tec.sample_encoding), _kv("Bit Rate", tec.bit_rate), _kv("Sample Rate", _hz_label(tec.sample_rate)), _kv("Channels", _channel_label(tec.channels)), _kv("Precision", tec.precision)]:
        if row: print(row)
    print(_section("DYNAMIC RANGE & LOUDNESS"))
    for line in _headroom_bar(lp.noise_floor_db, lp.rms_db, lp.peak_db): print(line)
    print(_subsection("Level Bookends"))
    for row in [_kv("Signal Ceiling", _c(_peak_colour(lp.peak_db), _db(lp.peak_db))), _kv("Noise Floor", _c(_noise_colour(lp.noise_floor_db), _db(lp.noise_floor_db))), _kv("RMS Loudness", _c(_rms_colour(lp.rms_db), _db(lp.rms_db))), _kv("RMS Peak", _db(lp.rms_peak_db)), _kv("RMS Trough", _db(lp.rms_trough_db))]:
        if row: print(row)
    print(_subsection("EBU R128"))
    for row in [_kv("LUFS Integrated", _c(_lufs_colour(lp.lufs_integrated), f"{lp.lufs_integrated} LUFS" if lp.lufs_integrated else "")), _kv("Loudness Range", f"{lp.lufs_range} LU" if lp.lufs_range else ""), _kv("True Peak", _c(_peak_colour(lp.true_peak_dbtp), f"{lp.true_peak_dbtp} dBTP" if lp.true_peak_dbtp else "")), _kv("Momentary Max", f"{lp.lufs_momentary_max} LUFS" if lp.lufs_momentary_max else ""), _kv("Short-term Max", f"{lp.lufs_shortterm_max} LUFS" if lp.lufs_shortterm_max else "")]:
        if row: print(row)
    print(_subsection("Streaming Normalization"))
    for row in [_kv("Apple Music (−16 LUFS)", _c(_delta_colour(lp.apple_music_delta), lp.apple_music_delta)), _kv("Spotify/Tidal (−14 LUFS)", _c(_delta_colour(lp.spotify_delta), lp.spotify_delta))]:
        if row: print(row)
    print(_subsection("Dynamic Quality"))
    dr_col, dr_desc = _dr_assessment(report.dr_score)
    for row in [_kv("DR Score (EBU)", _c(dr_col, f"{report.dr_score} — {dr_desc}")), _kv("DR (ffmpeg)", _db(lp.dynamic_range_db, " dB")), _kv("Crest Factor", _c(_crest_colour(lp.crest_factor_db), _db(lp.crest_factor_db, " dB") + " — compressed (modern standard)")), _kv("Flat Factor", _c(_flat_colour(lp.flat_factor), lp.flat_factor + (" — clean" if lp.flat_factor == "0.00" else " ⚠ limiting detected"))), _kv("SoX Entropy", _c(_sox_entropy_colour(lp.sox_entropy), lp.sox_entropy + " — " + _sox_entropy_interp(lp.sox_entropy)))]:
        if row: print(row)
    print(_subsection("Signal Integrity"))
    for row in [_kv("DC Offset", lp.dc_offset), _kv("Peak Events", lp.peak_count), _kv("Zero Crossing Rate", lp.zero_crossings_rate)]:
        if row: print(row)

    print(_section("AUTHENTICITY & FORENSICS"))
    print(_subsection("Spectral Analysis  (numpy FFT engine)"))
    sp = auth.spectral
    if sp and sp.verdict_label != "INCONCLUSIVE":
        conf_filled  = int(sp.net_confidence_pct / 10); conf_empty = 10 - conf_filled
        verdict_col  = {"GENUINE": C.GREEN, "LIKELY_GENUINE":C.GREEN, "CAUTION": C.YELLOW, "SUSPICIOUS": C.ORANGE, "LIKELY_LOSSY": C.RED}.get(sp.verdict_label, C.WHITE)
        conf_bar = _c(verdict_col, "█" * conf_filled) + _c(C.GREY, "░" * conf_empty)
        print(f"  {conf_bar} {_c(verdict_col + C.BOLD, sp.primary_verdict)}")
        print(f"  {_c(C.GREY, f'Raw Error Rate: {sp.raw_lossy_pct:.1f}%  |  Net Verdict Certainty: {sp.net_confidence_pct:.1f}%')}")
        print(f"  {_c(C.GREY, f'Score: Lossy {sp.lossy_score} − Natural {sp.natural_score} = Net {sp.net_score}/{sp.max_score}')}")
        print()
        rows_spec = [
            _kv("Ultrasonic Noise", _c(C.ORANGE, "⚠ DSD/SACD Transcode Profile") if sp.dsd_detected else _c(C.GREEN, "✓ Normal")),
            _kv("HF Cutoff",         sp.cutoff_hz_str),
            _kv("Cutoff Variance",   f"{sp.cutoff_variance:.1f} Hz²  " + _c(C.GREY, sp.cutoff_variance_interp)),
            _kv("Cliff Sharpness",   f"{sp.cutoff_sharpness_db:.1f} dB/bin  " + _c(C.GREY, sp.cutoff_sharpness_interp)),
            _kv("HF Energy Ratio",   f"{sp.hf_energy_ratio:.5f}  " + _c(C.GREY, sp.hf_energy_interp)),
            _kv("Side Anomaly",      f"{sp.side_anomaly_score:.3f}  " + _c(C.GREY, sp.side_interp)),
            _kv("Banding Score",     f"{sp.banding_score:.3f}  " + _c(C.GREY, sp.banding_interp)),
            _kv("NF Above Cutoff",   f"{sp.nf_above_cutoff_db:.1f} dB  " + _c(C.GREY, sp.nf_interp)),
            _kv("LPF",              ("⚠ YES — " + sp.lpf_cutoff_str) if sp.lpf_detected else "✓ None detected"),
            _kv("Spectral Entropy", f"{sp.entropy:.3f}  " + _c(C.GREY, sp.entropy_interp)),
        ]
        for row in rows_spec:
            if row: print(row)
        if sp.evidence:
            print(f"\n  {_c(C.DIM + C.ORANGE, 'Lossy indicators')}")
            for e in sp.evidence: print(f"    {_c(C.GREY, '·')} {_c(C.WHITE, e)}")
        if sp.natural_evidence:
            print(f"\n  {_c(C.DIM + C.GREEN, 'Natural indicators')}")
            for n in sp.natural_evidence: print(f"    {_c(C.GREY, '·')} {_c(C.GREEN, n)}")
        if sp.caveats:
            print(f"\n  {_c(C.DIM + C.GREY, 'Context notes')}")
            for cv in sp.caveats: print(f"    {_c(C.GREY, '·')} {_c(C.DIM + C.WHITE, cv)}")
    else:
        for row in [_kv("HF Cutoff", auth.spectral_cutoff_hz), _kv("Spectral Verdict", auth.spectral_cutoff_verdict), _kv("LPF Detected", ("⚠ YES — cutoff at " + auth.lpf_cutoff_hz) if auth.lpf_detected else "✓ No LPF detected")]:
            if row: print(row)

    print(_subsection("Source Integrity"))
    for row in [_kv("Bit-Depth Auth", auth.bit_depth_authentic), _kv("Phase Corr.", f"{auth.phase_correlation} {auth.phase_verdict}" if auth.phase_correlation else ""), _kv("Clipping", auth.clipping_verdict if auth.clipping_verdict else ""), _kv("Silence", auth.silence_total_pct)]:
        if row: print(row)
    if auth.silence_sections:
        for s in auth.silence_sections[:4]: print(f"    {_c(C.GREY, '→')} {_c(C.DIM + C.WHITE, s)}")
        if len(auth.silence_sections) > 4: print(f"    {_c(C.GREY, f'... +{len(auth.silence_sections)-4} more sections')}")

    print(_subsection("ReplayGain Audit"))
    if auth.rg_stored:
        for row in [_kv("RG Tag (stored)", auth.rg_stored), _kv("RG Measured", auth.rg_measured_lufs), _kv("Delta", auth.rg_delta), _kv("Verdict", auth.rg_verdict)]:
            if row: print(row)
    else: print(f"  {_c(C.GREY, 'No ReplayGain tags found')}")

    print(_section("ACOUSTIC MEASUREMENTS  (SoX)"))
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
            if val: rows.append(_kv(_fmt_stat_key(key), _c(_sox_amplitude_colour(key, val), val)))
        if rows:
            print(_subsection(gname))
            for row in rows: print(row)
    extras = [(k, v) for k, v in report.sox_stats.items() if k not in grouped]
    if extras:
        print(_subsection("Other"))
        for k, v in extras:
            if row := _kv(_fmt_stat_key(k), v): print(row)

    print()
    print(_rule("─", W))
    spec = report.spectrogram_path or report.filepath.with_name(f"{report.filepath.stem}_spectrogram.png")
    print(f"  {_c(C.GREEN,'✓')} Spectrogram → {_c(C.DIM + C.WHITE, str(spec))}")
    print(_rule("─", W))
    print()

def _report_to_dict(report: ForensicReport, file_size_mb: Optional[float] = None) -> dict:
    d = asdict(report)
    d["filepath"] = str(report.filepath)
    d["file_size_mb"] = file_size_mb if file_size_mb is not None else report.file_size_mb
    if report.spectrogram_path: d["spectrogram_path"] = str(report.spectrogram_path)
    return d

def print_batch_summary(reports: list[ForensicReport]) -> None:
    W = 78; print(); print(_rule("═", W)); print(f"  {_c(C.BOLD + C.WHITE, f'ALBUM BATCH  ·  {len(reports)} tracks')}"); print(_rule("═", W))
    col_w = [36, 6, 12, 10, 8]
    header = f"  {_c(C.GOLD, 'Track'.ljust(col_w[0]))} {_c(C.GOLD, 'DR'.ljust(col_w[1]))} {_c(C.GOLD, 'LUFS'.ljust(col_w[2]))} {_c(C.GOLD, 'NFloor'.ljust(col_w[3]))} {_c(C.GOLD, 'Verdict')}"
    print(header); print(_rule("─", W))
    for r in reports:
        name = r.filepath.name[:col_w[0]].ljust(col_w[0])
        dr = _c(_dr_assessment(r.dr_score)[0], r.dr_score.ljust(col_w[1]))
        lufs = r.loudness.lufs_integrated; lufs_s = _c(_lufs_colour(lufs), f"{lufs} LUFS".ljust(col_w[2]) if lufs else "---".ljust(col_w[2]))
        nf = r.loudness.noise_floor_db; nf_s = _c(_noise_colour(nf), f"{nf} dB".ljust(col_w[3]) if nf else "---".ljust(col_w[3]))
        verdict = r.authenticity.spectral_cutoff_verdict or "—"; vshort = verdict[:28]
        print(f"  {_c(C.WHITE, name)} {dr} {lufs_s} {nf_s} {_c(C.DIM + C.WHITE, vshort)}")
    print(_rule("─", W))
    drs = []
    for r in reports:
        try: drs.append((r.filepath.name, int(r.dr_score.replace("DR", ""))))
        except ValueError: pass
    if drs:
        avg_dr = sum(d for _, d in drs) / len(drs)
        outliers = [(n, d) for n, d in drs if abs(d - avg_dr) >= 3]
        if outliers:
            print(f"\n  {_c(C.YELLOW, '⚠ DR outliers (≥3 from album mean DR{:.0f}):'.format(avg_dr))}")
            for name, dr in outliers: print(f"    {_c(C.GREY, '→')} {name}  DR{dr}")
    print()

def main() -> None:
    parser = argparse.ArgumentParser(prog="audio_forensic", description="Audio Forensics CLI — comprehensive audio authenticity analysis")
    parser.add_argument("files", nargs="*", help="Audio file(s) to analyse")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--fast", action="store_true", help="Analyse first 60s only")
    parser.add_argument("--info", action="store_true", help="Only show basic metadata")
    args = parser.parse_args()

    missing = [t for t in ("ffmpeg", "sox", "mediainfo") if not _tool_available(t)]
    if missing:
        print(f"Error: Missing required tool(s): {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    if not args.files:
        parser.print_help(); sys.exit(1)

    paths = [Path(f) for f in args.files]
    missing_paths = [p for p in paths if not p.exists()]
    if missing_paths:
        for p in missing_paths: print(f"Error: not found — {p}", file=sys.stderr)
        sys.exit(1)

    if args.info:
        reports = [build_info_report(p) for p in paths]
        if args.json: print(json.dumps([_report_to_dict(r) for r in reports], indent=2, default=str))
        else:
            for report in reports: print_report(report)
        return

    reports = [build_report(p, fast_secs=60.0 if args.fast else None) for p in paths]
    if args.json:
        print(json.dumps([_report_to_dict(r) for r in reports], indent=2, default=str))
        return

    for report in reports: print_report(report)
    if len(reports) > 1: print_batch_summary(reports)

if __name__ == "__main__":
    main()
