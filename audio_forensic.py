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

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

try:
    from scipy import signal as _sps
    from scipy.fft import rfft as _srfft, rfftfreq as _srfftfreq
    from scipy.ndimage import median_filter as _ndi_median
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

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
    cliff_depth_db: float = 0.0
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
    # --- Advanced DSP suite (scipy) ---
    main_score: int = 0
    spectral_sparsity: float = 0.0; sparsity_interp: str = ""
    hf_envelope_correlation: float = 0.0; hf_env_corr_interp: str = ""
    preecho_pct: float = 0.0
    aliasing_corr: float = 0.0
    mp3_noise_pattern_detected: bool = False
    cassette_score: int = 0
    silence_ratio: float = -1.0
    vinyl_noise_detected: bool = False
    vinyl_clicks_per_min: float = 0.0
    header_duration_mismatch: bool = False
    header_bitrate_mismatch: bool = False
    segment_walled: int = -1; segment_total: int = 0; segment_wall_hz: float = 16500.0
    auc_avg_bound_freq: float = 0.0; auc_bound_interp: str = ""
    auc_prob_bound_freq: float = 0.0
    auc_phase_entropy: float = 0.0; auc_phase_interp: str = ""
    scipy_available: bool = True

@dataclass
class AuthenticityReport:
    spectral: "SpectralAnalysis | None" = None
    spectral_cutoff_hz: str = ""; spectral_cutoff_verdict: str = ""; lpf_detected: bool = False
    lpf_cutoff_hz: str = ""; bit_depth_authentic: str = ""; phase_correlation: str = ""
    phase_verdict: str = ""; clipped_samples: str = ""; clipping_verdict: str = ""
    silence_total_pct: str = ""; silence_sections: list[str] = field(default_factory=list)
    rg_stored: str = ""; rg_measured_lufs: str = ""; rg_delta: str = ""; rg_verdict: str = ""
    cassette_rip_detected: bool = False
    vinyl_rip_detected: bool = False
    side_channel_analysis: str = ""
    header_integrity: str = ""

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
# Advanced DSP helpers (scipy-backed; engine degrades gracefully without them)
# ---------------------------------------------------------------------------
def bandpass_filter(data: "np.ndarray", lowcut: float, highcut: float, fs: int, order: int = 4) -> "np.ndarray":
    """Apply Butterworth bandpass filter to a 1D NumPy array."""
    sos = _sps.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return _sps.sosfilt(sos, data)

def highpass_filter(data: "np.ndarray", cutoff: float, fs: int, order: int = 4) -> "np.ndarray":
    """Apply Butterworth highpass filter."""
    sos = _sps.butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return _sps.sosfilt(sos, data)

def calculate_autocorrelation(data: "np.ndarray", lag: int = 50) -> float:
    """Normalized absolute autocorrelation at a given sample lag (0 = random noise, 1 = periodic)."""
    if len(data) <= lag * 2: return 0.0
    segment = data - np.mean(data)
    std = np.std(segment)
    if std < 1e-10: return 0.0
    segment = segment / std
    corr = np.corrcoef(segment[:-lag], segment[lag:])[0, 1]
    return float(np.abs(corr)) if not np.isnan(corr) else 0.0

def calculate_temporal_variance(data: "np.ndarray", sample_rate: int, segment_duration: float = 1.0) -> float:
    """Standard deviation of per-segment RMS energy in dB over time (tape hiss is very stable)."""
    seg_samples = int(segment_duration * sample_rate)
    num_segs = len(data) // seg_samples
    if num_segs < 2: return 0.0
    segs = data[: num_segs * seg_samples].reshape(num_segs, seg_samples)
    rms = np.sqrt(np.mean(segs ** 2, axis=1))
    energies_db = 20 * np.log10(rms + 1e-12)
    return float(np.std(energies_db))

# ---------------------------------------------------------------------------
# SpectralEngine — numpy/scipy FFT-based authenticity analysis
# ---------------------------------------------------------------------------

class SpectralEngine:
    WINDOW = 4096; HOP = 2048; CUTOFF_DB = -65.0; NYQUIST_MARGIN = 0.85
    SCORE_CUTOFF_WELL_BELOW_NYQUIST = 2; SCORE_SHARP_CLIFF_HARD = 3; SCORE_SHARP_CLIFF_SOFT = 1
    SCORE_HF_NEAR_ZERO = 1; SCORE_VOID_ABOVE_CUTOFF = 3; SCORE_QUIET_ABOVE_CUTOFF = 1
    SCORE_VERY_STABLE_CUTOFF = 1; SCORE_BANDING_STRONG = 1; SCORE_SIDE_ANOMALY = 2
    MAX_LOSSY_SCORE = 14
    NATURAL_GRADUAL_ROLLOFF = 1; NATURAL_HIGH_VARIANCE = 1; NATURAL_MODERATE_VARIANCE = 1
    NATURAL_RICH_HF = 1; NATURAL_HF_NOISE = 1; NATURAL_HEALTHY_SIDE = 1; NATURAL_HIGH_ENTROPY = 1
    # Empirically measured LAME lowpass cutoffs (-65 dB point, 95th percentile per frame)
    MP3_CUTOFFS = {320: 20200, 256: 19550, 224: 19550, 192: 18850, 160: 17450, 128: 16800, 96: 15400, 64: 11100}

    # Time-domain analyses (Hilbert envelopes, cascaded band filters) are capped to
    # this many seconds to bound CPU/RAM on very long files; spectral stats use the full decode.
    TIME_DOMAIN_CAP_S = 180.0

    def __init__(self, filepath: Path, sample_rate: int, channels: int = 2,
                 claimed_duration: float = 0.0, claimed_bitrate_kbps: int = 0):
        self.filepath = filepath; self.sample_rate = sample_rate; self.nyquist = sample_rate / 2.0
        self.channels = channels
        self.claimed_duration = claimed_duration
        self.claimed_bitrate_kbps = claimed_bitrate_kbps

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
        return self._compute_stft(audio)[0]

    def _compute_stft(self, audio: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, int]":
        """Vectorized chunked STFT. Returns (magnitude [frames, bins] float32,
        phase of bins >= 10 kHz [frames, hi_bins] float32, index of first hi bin).
        Magnitude feeds every spectral detector; high-band phase feeds auCDtect entropy."""
        n_frames = max(0, (len(audio) - self.WINDOW + self.HOP - 1) // self.HOP)  # == len(range(0, len-WINDOW, HOP))
        win = np.hanning(self.WINDOW).astype(np.float32)
        bins = self._freq_bins()
        bin_hz = bins[1] - bins[0]
        hi_start = min(len(bins) - 1, int(10000 / bin_hz))
        idx = np.arange(self.WINDOW)
        mags, phases = [], []
        CHUNK = 256
        for start in range(0, n_frames, CHUNK):
            cnt = min(CHUNK, n_frames - start)
            offs = (np.arange(cnt) + start) * self.HOP
            block = audio[offs[:, None] + idx[None, :]] * win
            spec = np.fft.rfft(block, axis=1)
            mags.append(np.abs(spec).astype(np.float32))
            phases.append(np.angle(spec[:, hi_start:]).astype(np.float32))
        if not mags:
            empty = np.zeros((0, len(bins)), dtype=np.float32)
            return empty, empty[:, hi_start:], hi_start
        return np.concatenate(mags), np.concatenate(phases), hi_start

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

    def _interp_bound(self, avg_bound: float) -> str:
        if avg_bound <= 0: return ""
        if avg_bound >= self.nyquist * 0.85: return "[organic scatter to the ceiling: lossless-like]"
        elif avg_bound >= 16500: return "[moderate bound: high-bitrate encode or dark master]"
        else: return "[scatter collapse: statistical void left by a lossy codec]"

    @staticmethod
    def _interp_phase_entropy(e: float, legit_cutoff: bool) -> str:
        if e <= 0: return ""
        if e < 4.0: return "[structured HF phase: tonal/organic]"
        elif e < 4.5: return "[typical phase complexity]"
        elif legit_cutoff: return "[high but full-spectrum: dither/noise content]"
        else: return "[quantized high-band phase: codec disruption]"

    @staticmethod
    def _interp_sparsity(s: float, legit_cutoff: bool) -> str:
        if s < 0.05: return "[dense spectrum: no psychoacoustic holes]"
        elif s < 0.30: return "[some quiet bins: normal for dynamic audio]"
        elif legit_cutoff: return "[sparse but full-bandwidth: very dynamic content]"
        else: return "[psychoacoustic holes below cutoff: codec bin-zeroing]"

    @staticmethod
    def _interp_ultra_corr(c: float) -> str:
        if c > 0.6: return "[HF breathes with the music: genuine harmonics]"
        elif c > 0.3: return "[moderate coupling: normal]"
        elif c > 0.15: return "[weak coupling: noisy or dark HF]"
        else: return "[HF independent of music: dither, hiss, or injected fake noise]"

    @staticmethod
    def _active_frame_mask(frames: "np.ndarray") -> "np.ndarray":
        """Mask of non-silent frames (peak within 60 dB of the loudest frame).
        Silent passages have no spectral content and poison cutoff/bound statistics."""
        peaks = frames.max(axis=1)
        ref = peaks.max() + 1e-12
        return peaks > ref * 1e-3  # -60 dB

    def _cutoff_per_frame(self, frames: "np.ndarray", bins: "np.ndarray") -> "np.ndarray":
        if frames.shape[0] == 0: return np.zeros(0)
        ref = frames.max(axis=1, keepdims=True) + 1e-12
        db = 20.0 * np.log10(frames / ref + 1e-12)
        mask = db > self.CUTOFF_DB
        has_any = mask.any(axis=1)
        last_idx = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
        return np.where(has_any, bins[last_idx], 0.0)

    def _sharpness(self, frames: "np.ndarray", bins: "np.ndarray", cutoff_hz: float, window_hz: float = 2500.0) -> float:
        bin_hz = bins[1] - bins[0]
        lo = max(0, int((cutoff_hz - window_hz) / bin_hz))
        hi = min(len(bins), int((cutoff_hz + window_hz * 0.25) / bin_hz))
        avg = frames.mean(axis=0)
        db = 20.0 * np.log10(avg[lo:hi] / (avg.max() + 1e-12) + 1e-12)
        return float(np.abs(np.diff(db)).max()) if len(db) > 1 else 0.0

    def _cliff_depth(self, frames: "np.ndarray", bins: "np.ndarray", cutoff_hz: float, span_hz: float = 400.0) -> float:
        """dB drop across ±span_hz around the cutoff. A codec wall falls 35+ dB inside
        800 Hz; natural rolloff loses a few dB. Complements the per-bin gradient, which
        under-reads walls whose transition spans dozens of bins."""
        bin_hz = bins[1] - bins[0]
        avg = frames.mean(axis=0)
        ref = avg.max() + 1e-12
        lo = int(max(0, (cutoff_hz - span_hz) / bin_hz))
        hi = int(min(len(bins) - 1, (cutoff_hz + span_hz) / bin_hz))
        if hi <= lo: return 0.0
        db = 20.0 * np.log10(avg / ref + 1e-12)
        return float(db[lo] - db[hi])

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

    def _side_channel_anomaly(self, mid_frames: "np.ndarray", side: "np.ndarray", bins: "np.ndarray") -> float:
        """Joint-stereo forensics on y_side = (L−R)/2 — codecs starve the side channel of HF first."""
        if not _NUMPY_OK or side is None or len(side) < self.WINDOW * 2: return 0.0
        side_frames = self._compute_frames(side)

        bin_hz = bins[1] - bins[0]
        idx_10k = int(10000 / bin_hz)
        if idx_10k >= mid_frames.shape[1]: return 0.0

        n = min(mid_frames.shape[0], side_frames.shape[0])
        mid_hf = mid_frames[:n, idx_10k:]
        side_hf = side_frames[:n, idx_10k:]

        e_ratio = float(np.mean(side_hf)) / (float(np.mean(mid_hf)) + 1e-12)
        score, wt = 0.0, 1.0
        if e_ratio < 0.02: score += 1.0
        elif e_ratio < 0.08: score += 0.6
        return float(score / wt)

    # -----------------------------------------------------------------------
    # auCDtect-style statistical analysis (bound frequency + high-band phase)
    # -----------------------------------------------------------------------
    def _aucdtect_features(self, frames: "np.ndarray", phase_hi: "np.ndarray", bins: "np.ndarray") -> tuple[float, float, float]:
        """Returns (avg_bound_freq, most_probable_bound_freq, high_band_phase_entropy).

        Bound frequency: per frame, the spectral 'scatter' (5-bin sliding std of log power)
        stays organic (>0.6) wherever real signal/dither lives and collapses to ~0 in the
        digitally voided region a lossy codec leaves behind. Bins more than 110 dB below the
        frame peak are clamped first so numerical decoder residue reads as a true void.
        Robust against flat noise injection — uniform fake noise has near-zero scatter too.
        """
        if frames.shape[0] < 4 or frames.shape[1] < 10:
            return 0.0, 0.0, 0.0
        ref = frames.max(axis=1, keepdims=True) + 1e-12
        db = 20.0 * np.log10(frames / ref + 1e-12)
        db = np.maximum(db, -110.0)               # clamp: decoder numerical residue -> constant
        log_power = (db / 10.0) * np.log(10.0)    # back to natural-log power scale

        sw = np.lib.stride_tricks.sliding_window_view(log_power, 5, axis=1)
        scatter = sw.std(axis=-1)
        if _SCIPY_OK:
            scatter = _ndi_median(scatter, size=(1, 5), mode="nearest")

        max_sc = scatter.max(axis=1, keepdims=True)
        thresh = np.minimum(0.6, max_sc * 0.25)
        organic = scatter >= thresh
        has_any = organic.any(axis=1)
        last_idx = organic.shape[1] - 1 - np.argmax(organic[:, ::-1], axis=1)
        bound_bins = np.where(has_any, last_idx + 2, 0)   # +2: sliding-window centre offset
        bound_freqs = bins[np.minimum(bound_bins, len(bins) - 1)]

        avg_bound = float(np.mean(bound_freqs))
        hist, edges = np.histogram(bound_freqs, bins=min(20, frames.shape[1]))
        mi = int(np.argmax(hist))
        prob_bound = float((edges[mi] + edges[mi + 1]) / 2.0)

        # High-band (>=10 kHz) phase-difference entropy: lossy codecs randomize HF phase.
        phase_entropy = 0.0
        if phase_hi.shape[0] >= 3 and phase_hi.shape[1] >= 2:
            pd = np.diff(phase_hi.astype(np.float64), axis=0)
            pdw = np.arctan2(np.sin(pd), np.cos(pd))
            hist_p, _ = np.histogram(pdw, bins=36, range=(-np.pi, np.pi))
            p = hist_p / (hist_p.sum() + 1e-12)
            p = p[p > 0]
            phase_entropy = float(-np.sum(p * np.log2(p)))

        return avg_bound, prob_bound, phase_entropy

    # -----------------------------------------------------------------------
    # Fakin' the Funk — header integrity (duration & bitrate plausibility)
    # -----------------------------------------------------------------------
    def _check_header_integrity(self, decoded_duration: float) -> tuple[bool, bool, list[str]]:
        duration_mismatch, bitrate_mismatch, reasons = False, False, []
        if self.claimed_duration > 0 and decoded_duration > 0:
            diff = abs(decoded_duration - self.claimed_duration)
            if diff > 0.5:
                duration_mismatch = True
                reasons.append(f"Header Mismatch: Container claims {self.claimed_duration:.2f}s but the frame decoder yields {decoded_duration:.2f}s (Δ {diff:.2f}s) — header has been forged or the stream is truncated.")
        ext = self.filepath.suffix.lower()
        if self.claimed_bitrate_kbps > 0 and self.claimed_duration > 1.0 and ext in {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}:
            try:
                actual_kbps = self.filepath.stat().st_size * 8.0 / self.claimed_duration / 1000.0
                if actual_kbps < self.claimed_bitrate_kbps * 0.80 or actual_kbps > self.claimed_bitrate_kbps * 1.35:
                    bitrate_mismatch = True
                    reasons.append(f"Bitrate Forgery: Header claims {self.claimed_bitrate_kbps} kbps but file size implies ~{actual_kbps:.0f} kbps of actual payload.")
            except OSError:
                pass
        return duration_mismatch, bitrate_mismatch, reasons

    # -----------------------------------------------------------------------
    # Audio Fake Detector PRO — N-segment wall-check voting
    # -----------------------------------------------------------------------
    def _segment_voting(self, audio: "np.ndarray", n_segments: int = 7, wall_hz: float = 16500.0) -> tuple[int, int, bool]:
        """Cutoff-walls N spread-out 2 s clips and takes a majority vote.
        Catches files where only parts were transcoded (spliced fakes).
        wall_hz is adaptive: when the global cutoff has a verified digital void above it,
        the wall threshold tracks that cutoff instead of the fixed AFD 16.5 kHz."""
        if self.nyquist <= wall_hz: return -1, 0, False
        total = len(audio)
        seg_samples = int(2.0 * self.sample_rate)
        if total < seg_samples * n_segments: return -1, 0, False

        import random
        rng = random.Random(42)  # deterministic
        offsets = []
        step = (total - seg_samples) // (n_segments - 1)
        for i in range(n_segments - 2):
            offsets.append(i * step)
        offsets.append(rng.randint(0, total - seg_samples))
        offsets.append(total - seg_samples)

        walled = 0
        win = np.hanning(seg_samples)
        freqs = np.fft.rfftfreq(seg_samples, 1.0 / self.sample_rate)
        for off in offsets:
            clip = audio[off : off + seg_samples].astype(np.float64)
            mag = np.abs(np.fft.rfft(clip * win))
            ref = mag.max() + 1e-12
            db = 20 * np.log10(mag / ref + 1e-12)
            above = np.where(db > self.CUTOFF_DB)[0]
            cutoff = freqs[above[-1]] if len(above) else 0.0
            if cutoff <= wall_hz:
                walled += 1

        if n_segments % 2 == 0: is_fake = walled >= (n_segments / 2)
        else: is_fake = walled > (n_segments / 2)
        return walled, n_segments, bool(is_fake)

    def _fft_band_extract(self, x: "np.ndarray", lo: float, hi: float) -> "np.ndarray":
        """Zero-phase brickwall band extraction via FFT masking. IIR skirts (~24 dB/oct)
        leak loud music into a quiet band only ~0.1 octave away; spectral masking gives
        total rejection, which noise-floor forensics above the cutoff depend on."""
        X = np.fft.rfft(x.astype(np.float64))
        f = np.fft.rfftfreq(len(x), 1.0 / self.sample_rate)
        X[(f < lo) | (f > hi)] = 0.0
        return np.fft.irfft(X, n=len(x))

    # -----------------------------------------------------------------------
    # 3-Phase silence / dither / vinyl-surface-noise analyser
    # -----------------------------------------------------------------------
    def _silence_and_vinyl(self, audio: "np.ndarray", cutoff_hz: float, noise_band: "np.ndarray | None" = None) -> tuple[int, list[str], float, bool, float]:
        """Phase 1: dither energy ratio inside silent passages (codec noise vs clean dither).
        Phase 2: noise floor character above the cutoff (vinyl hiss is random & stable).
        Phase 3: click/pop transient counting to confirm a vinyl source."""
        score, reasons = 0, []
        silence_ratio, vinyl_detected, clicks_per_min = -1.0, False, 0.0
        sr = self.sample_rate

        # --- Phase 1: silence dither ratio
        threshold_linear = 10 ** (-40.0 / 20.0)
        is_sil = np.abs(audio) < threshold_linear
        padded = np.concatenate(([False], is_sil, [False]))
        d = np.diff(padded.astype(np.int8))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        min_samples = int(0.5 * sr)
        sil_segs = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_samples]
        total_sil_sec = sum(e - s for s, e in sil_segs) / sr

        if total_sil_sec >= 2.0:
            if len(audio) >= int(40 * sr): music_ref = audio[int(10 * sr):int(40 * sr)]
            else: music_ref = audio
            sil_cap = int(30 * sr)
            silence_ref = np.concatenate([audio[s:e] for s, e in sil_segs])[:sil_cap]
            upper_limit = min(22000.0, sr / 2 - 100)

            def hf_energy(segment: "np.ndarray") -> float:
                # Normalize by N² so the ratio is invariant to segment length
                # (Parseval: band sum of |X|² scales with N²·power).
                if len(segment) < 1024: return 0.0
                win = np.hanning(len(segment))
                fft_res = np.abs(np.fft.rfft(segment * win)) ** 2
                freqs = np.fft.rfftfreq(len(segment), 1.0 / sr)
                idx = (freqs >= 16000) & (freqs <= upper_limit)
                return float(np.sum(fft_res[idx]) / len(segment) ** 2) if np.any(idx) else 0.0

            e_music, e_silence = hf_energy(music_ref), hf_energy(silence_ref)
            if e_music > 0:
                silence_ratio = e_silence / (e_music + 1e-12)
                if silence_ratio > 0.3:
                    score += 50
                    reasons.append(f"Codec Noise in Silence: silent passages carry {silence_ratio:.2f}× the music's ultrasonic energy — artificial dither/codec hash, not clean studio silence.")
                    return score, reasons, silence_ratio, False, 0.0
                elif silence_ratio < 0.15:
                    score -= 50
                    reasons.append(f"Clean Silence Floor: silent passages are spectrally clean (ratio {silence_ratio:.2f}) — consistent with an unmolested lossless master.")
                    return score, reasons, silence_ratio, False, 0.0

        # --- Phase 2: vinyl surface noise above the cutoff
        # Band starts 1 kHz above the detected cutoff; FFT brickwall extraction so
        # music-band energy cannot leak in and masquerade as a noise floor.
        cap = audio[: int(self.TIME_DOMAIN_CAP_S * sr)]
        if noise_band is not None or 0 < cutoff_hz < self.nyquist - 2100:
            if noise_band is None:
                noise_band = self._fft_band_extract(cap, cutoff_hz + 1000, self.nyquist - 100)
            rms = float(np.sqrt(np.mean(noise_band ** 2)))
            energy_db = 20 * math.log10(rms + 1e-12)
            if energy_db < -70.0:
                score += 20
                reasons.append(f"Digital Upscale Suspect: no noise floor above the cutoff ({energy_db:.1f} dB) — analog sources always leave hiss there.")
            else:
                autocorr = calculate_autocorrelation(noise_band, lag=50)
                variance = calculate_temporal_variance(noise_band, sr)
                if autocorr < 0.3 and variance < 5.0:
                    vinyl_detected = True
                    score -= 40
                    reasons.append(f"Vinyl Surface Noise: random ({autocorr:.2f} autocorr), temporally stable hiss above the cutoff ({energy_db:.1f} dB) — analog playback signature.")

                    # --- Phase 3: clicks & pops
                    hp = highpass_filter(cap.astype(np.float64), 1000, sr)
                    env = np.abs(_sps.hilbert(hp, N=len(hp)))
                    k = int(0.0005 * sr) | 1
                    env_smooth = _ndi_median(env, size=k, mode="nearest")
                    peaks, _ = _sps.find_peaks(env_smooth, height=float(np.median(env_smooth)) * 3, distance=int(0.01 * sr))
                    clicks_per_min = (len(peaks) / (len(cap) / sr)) * 60
                    if 5 <= clicks_per_min <= 50:
                        score -= 10
                        reasons.append(f"Vinyl Clicks Confirmed: {clicks_per_min:.1f} click transients/min — physical media artefacts.")

        return score, reasons, silence_ratio, vinyl_detected, clicks_per_min

    # -----------------------------------------------------------------------
    # Psychoacoustic artefacts — pre-echo, HF aliasing, MP3 subband comb
    # -----------------------------------------------------------------------
    def _psychoacoustic_artifacts(self, audio: "np.ndarray", frames: "np.ndarray", bins: "np.ndarray", cutoff_hz: float, mp3_detected: bool) -> tuple[int, list[str], float, float, bool]:
        score, reasons = 0, []
        preecho_pct, aliasing_corr, mp3_noise_pattern = 0.0, 0.0, False
        sr = self.sample_rate
        if cutoff_hz >= 21000 and not mp3_detected:
            return score, reasons, 0.0, 0.0, False

        cap = audio[: int(self.TIME_DOMAIN_CAP_S * sr)].astype(np.float64)

        # 9A: Pre-echo — MDCT block smearing leaks HF energy *before* sharp transients
        env = np.abs(_sps.hilbert(cap, N=len(cap)))
        k = int(0.001 * sr) | 1
        env_smooth = _ndi_median(env, size=k, mode="nearest")
        peaks, _ = _sps.find_peaks(env_smooth, height=10 ** (-3.0 / 20.0), distance=int(0.05 * sr))
        if len(peaks) > 0:
            hf = bandpass_filter(cap, 10000, min(20000, self.nyquist - 100), sr)
            baseline = float(np.median(hf ** 2))
            pre_w, post_w = int(0.02 * sr), int(0.01 * sr)
            affected = 0
            for p in peaks:
                if p < pre_w + post_w: continue
                pre_energy = float(np.mean(hf[p - pre_w : p - post_w] ** 2))
                if pre_energy > baseline * 3: affected += 1
            preecho_pct = (affected / len(peaks)) * 100
            if preecho_pct > 10:
                score += 15
                reasons.append(f"Pre-Echo Smearing: {preecho_pct:.1f}% of transients show HF energy bleeding backwards in time — MDCT block artefact.")
            elif preecho_pct >= 5:
                score += 10
                reasons.append(f"Moderate Pre-Echo: {preecho_pct:.1f}% of transients show pre-transient HF leakage.")

        # 9B: HF aliasing — filterbank aliasing mirrors 10-15k into inverted 15-20k
        if self.nyquist >= 15000:
            band_a = bandpass_filter(cap, 10000, 15000, sr)
            band_b_inv = -bandpass_filter(cap, 15000, min(20000, self.nyquist - 100), sr)
            seg_len = min(len(band_a), int(sr * 5))
            corrs = []
            for i in range(0, len(band_a) - seg_len + 1, max(1, seg_len // 2)):
                sa, sb = band_a[i : i + seg_len], band_b_inv[i : i + seg_len]
                if np.std(sa) > 1e-6 and np.std(sb) > 1e-6:
                    c = np.abs(np.corrcoef(sa, sb)[0, 1])
                    if not np.isnan(c): corrs.append(float(c))
            aliasing_corr = float(np.median(corrs)) if corrs else 0.0
            if aliasing_corr > 0.5:
                score += 15
                reasons.append(f"Severe Filterbank Aliasing: mirrored HF correlation {aliasing_corr:.2f} — codec synthesis artefact.")
            elif aliasing_corr >= 0.3:
                score += 10
                reasons.append(f"Moderate Filterbank Aliasing: mirrored HF correlation {aliasing_corr:.2f}.")

        # 9C: MP3 subband comb — 32-band filterbank leaves spectral peaks every 689.06 Hz
        if self.nyquist >= 16000 and frames.shape[0] >= 4:
            bin_hz = bins[1] - bins[0]
            lo, hi = int(16000 / bin_hz), min(frames.shape[1], int(min(20000.0, self.nyquist - 100) / bin_hz))
            if hi - lo > 80:
                avg_db = 20.0 * np.log10(frames[:, lo:hi].mean(axis=0) + 1e-12)
                spec = avg_db - avg_db.mean()
                denom = float(np.sum(spec ** 2)) + 1e-12
                subband_hz = self.sample_rate / 64.0  # MP3 subband width (689.06 Hz @ 44.1k)
                peaks_found = 0
                for mult in (1, 2, 3):
                    lag = int(round(mult * subband_hz / bin_hz))
                    if lag <= 0 or lag >= len(spec) - 1: continue
                    ac = float(np.sum(spec[:-lag] * spec[lag:])) / denom
                    neighbours = []
                    for nl in (lag - 3, lag + 3):
                        if 0 < nl < len(spec) - 1:
                            neighbours.append(abs(float(np.sum(spec[:-nl] * spec[nl:])) / denom))
                    if ac > 0.25 and (not neighbours or ac > 2 * max(neighbours)):
                        peaks_found += 1
                if peaks_found >= 2:
                    mp3_noise_pattern = True
                    score += 10
                    reasons.append("MP3 Subband Comb: periodic spectral structure at 689 Hz multiples — 32-band filterbank residue.")

        return score, reasons, preecho_pct, aliasing_corr, mp3_noise_pattern

    # -----------------------------------------------------------------------
    # Rule 11 — analogue cassette source profiler (false-positive bypass)
    # -----------------------------------------------------------------------
    def _cassette_source(self, audio: "np.ndarray", cutoff_hz: float, cutoff_std: float, mp3_detected: bool) -> tuple[int, list[str]]:
        score, reasons = 0, []
        if cutoff_hz >= 19000: return 0, reasons
        sr = self.sample_rate
        cap = audio[: int(60.0 * sr)].astype(np.float64)

        # 11A: constant tape hiss above the musical cutoff (FFT brickwall — no music leakage)
        upper_limit = min(20000.0, sr / 2 - 100)
        noise_lo = cutoff_hz + 1000 if cutoff_hz < 16000 else cutoff_hz + 500
        if upper_limit > noise_lo:
            noise_sig = self._fft_band_extract(cap, noise_lo, upper_limit)
            noise_db = 20 * math.log10(float(np.std(noise_sig)) + 1e-12)
            autocorr = calculate_autocorrelation(noise_sig, lag=100)
            if noise_db > -55.0 and autocorr < 0.2:
                score += 30
                reasons.append(f"R11A: Tape hiss present above cutoff ({noise_db:.1f} dB, random autocorr {autocorr:.2f}).")

        # 11B: natural magnetic-tape roll-off slope across 12-18 kHz
        freqs = np.linspace(12000, 18000, 20)
        res = []
        for f in freqs:
            if f + 250 < sr / 2:
                band_sig = bandpass_filter(cap, f - 250, f + 250, sr)
                res.append(20 * math.log10(float(np.std(band_sig)) + 1e-12))
            else:
                res.append(-120.0)
        slope = (res[-1] - res[0]) / 6.0
        if -6.0 < slope < -3.0:
            score += 20
            reasons.append(f"R11B: Natural tape roll-off slope ({slope:.1f} dB/kHz) — gradual analog decay, not a brick wall.")
        elif slope < -10.0:
            score -= 20

        # 11C: no codec filterbank artefacts
        if not mp3_detected:
            score += 15
            reasons.append("R11C: No codec subband artefacts found.")

        # 11D: wow/flutter — tape speed instability modulates the cutoff
        if 50 < cutoff_std < 300:
            score += 15
            reasons.append(f"R11D: Wow/flutter spectral modulation (cutoff σ {cutoff_std:.0f} Hz).")
        elif cutoff_std < 30:
            score -= 10

        return max(0, score), reasons

    # -----------------------------------------------------------------------
    # Spectral sparsity & ultrasonic envelope correlation (anti-forensics)
    # -----------------------------------------------------------------------
    def _spectral_sparsity(self, frames: "np.ndarray", bins: "np.ndarray", cutoff_hz: float) -> float:
        """Fraction of psychoacoustically zeroed bins (<-95 dB rel.) BELOW the cutoff —
        codecs punch holes in the audible band that no natural recording has."""
        bin_hz = bins[1] - bins[0]
        cutoff_idx = min(frames.shape[1], int(cutoff_hz / bin_hz))
        if cutoff_idx < 10 or frames.shape[0] == 0: return 0.0
        region = frames[:, :cutoff_idx]
        ref = frames.max(axis=1, keepdims=True) + 1e-12
        db = 20.0 * np.log10(region / ref + 1e-12)
        return float(np.sum(db < -95.0) / (db.size + 1e-12))

    def _ultrasonic_envelope_correlation(self, frames: "np.ndarray", bins: "np.ndarray") -> float:
        """Pearson correlation between the mid-band (1-8 kHz) and high-band (16-22 kHz)
        energy envelopes. Genuine HF content breathes with the music; injected fake
        ultrasonic noise (anti-forensic masking) is statistically independent of it."""
        if frames.shape[0] < 10: return 1.0
        bin_hz = bins[1] - bins[0]
        m_lo, m_hi = int(1000 / bin_hz), int(8000 / bin_hz)
        h_lo, h_hi = int(16000 / bin_hz), min(frames.shape[1], int(22000 / bin_hz))
        if h_hi <= h_lo or m_hi <= m_lo: return 1.0
        env_mid = np.sqrt(np.mean(frames[:, m_lo:m_hi].astype(np.float64) ** 2, axis=1))
        env_high = np.sqrt(np.mean(frames[:, h_lo:h_hi].astype(np.float64) ** 2, axis=1))
        std_mid, std_high = float(np.std(env_mid)), float(np.std(env_high))
        if std_mid < 1e-8 or std_high < 1e-8: return 0.0
        corr = float(np.mean((env_mid - env_mid.mean()) * (env_high - env_high.mean())) / (std_mid * std_high))
        return corr

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

    def _score(self, cutoff_hz: float, variance: float, sharpness: float, cliff_depth: float, hf_ratio: float, nf_above: float, banding: float, side_anomaly: float, entropy: float, dsd_detected: bool) -> tuple[int, list[str], int, list[str]]:
        l_score, l_ev, n_score, n_ev = 0, [], 0, []

        if cutoff_hz < self.nyquist * 0.85 and cutoff_hz < 18500:
            l_score += self.SCORE_CUTOFF_WELL_BELOW_NYQUIST
            l_ev.append(f"Premature High-Frequency Rolloff: Hard cutoff detected at {cutoff_hz:,.0f} Hz, strongly suggesting lossy perceptual encoding.")
        # Cliff depth only counts below 93% of Nyquist — mastering-grade SRC brickwalls
        # (e.g. 48k->44.1k conversion) legitimately live above that.
        deep_cliff = cliff_depth > 35.0 and cutoff_hz < self.nyquist * 0.93
        moderate_cliff = cliff_depth > 20.0 and cutoff_hz < self.nyquist * 0.93
        if sharpness > 15.0 or deep_cliff:
            l_score += self.SCORE_SHARP_CLIFF_HARD
            l_ev.append(f"Unnatural Spectral Cliff: Spectrum falls {cliff_depth:.0f} dB across 800 Hz at the ceiling ({sharpness:.1f} dB/bin gradient) — an algorithmic low-pass wall.")
        elif sharpness > 8.0 or moderate_cliff:
            l_score += self.SCORE_SHARP_CLIFF_SOFT
            l_ev.append(f"Steep Frequency Ceiling: Substantial spectral cliff detected ({cliff_depth:.0f} dB across 800 Hz, {sharpness:.1f} dB/bin).")
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

    def _verdict(self, main_score: int, net_score: int, cutoff_hz: float, dsd_detected: bool,
                 cassette: bool = False, vinyl: bool = False) -> tuple[str, str, list[str]]:
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
        if not _SCIPY_OK: caveats.append("scipy not installed — advanced DSP suite skipped; verdict relies on the base spectral engine only.")
        if dsd_detected: caveats.append("DSD transcode detected. Ultrasonic noise inflates entropy and HF scores.")
        if cassette: caveats.append("Cassette source profile matched — HF limitations are analog tape physics, not codec damage.")
        if vinyl: caveats.append("Vinyl surface noise detected — rolloff and noise floor traits are analog, not codec damage.")
        if main_score >= 86: return "LIKELY_LOSSY", "✗  Lossy transcode detected — fake lossless (high certainty)", caveats
        elif main_score >= 55: return "SUSPICIOUS", "⚠  Strong lossy indicators — probable transcode", caveats
        elif main_score >= 31: return "CAUTION", "~  Minor spectral quirks — possibly legitimate", caveats
        elif main_score >= 11: return "LIKELY_GENUINE", "✓  Consistent with genuine lossless source", caveats
        else: return "GENUINE", "✓  Strong evidence of authentic lossless source", caveats

    def analyse(self, max_seconds: Optional[float] = None) -> SpectralAnalysis:
        result = SpectralAnalysis()
        result.scipy_available = _SCIPY_OK
        if not _NUMPY_OK:
            result.primary_verdict = "numpy not installed"; result.verdict_label = "INCONCLUSIVE"; return result
        if not _SCIPY_OK:
            print("Warning: scipy not installed — advanced forensic suite (auCDtect, vinyl/cassette, "
                  "psychoacoustic tests) disabled. pip install scipy", file=sys.stderr)

        # Single decode: stereo when available (mid feeds every mono detector, side feeds joint-stereo forensics)
        audio, side = None, None
        if self.channels >= 2 and (pair := self._decode_stereo(max_seconds)) is not None:
            audio, side = pair
        if audio is None:
            audio = self._decode_audio(max_seconds)
        if audio is None or len(audio) < self.WINDOW * 2:
            result.primary_verdict = "Could not decode audio"; result.verdict_label = "INCONCLUSIVE"; return result

        frames_all, phase_hi, _ = self._compute_stft(audio)
        bins = self._freq_bins()
        if frames_all.shape[0] < 4:
            result.primary_verdict = "File too short"; result.verdict_label = "INCONCLUSIVE"; return result

        # Silent frames carry no spectral evidence — exclude them from all statistics
        active = self._active_frame_mask(frames_all)
        if int(active.sum()) >= 4:
            frames, phase_act = frames_all[active], phase_hi[active]
        else:
            frames, phase_act = frames_all, phase_hi

        cutoffs_per_frame = self._cutoff_per_frame(frames, bins)
        cutoff_hz, cutoff_var = float(np.percentile(cutoffs_per_frame, 95)), float(np.var(cutoffs_per_frame))
        cutoff_std = math.sqrt(cutoff_var)
        sharpness, hf_ratio = self._sharpness(frames, bins, cutoff_hz), self._hf_energy_ratio(frames, bins)
        cliff_depth = self._cliff_depth(frames, bins, cutoff_hz)
        banding, nf_above = self._banding_score(frames, bins, cutoff_hz), self._noise_floor_above_cutoff(frames, bins, cutoff_hz)
        lpf_detected, lpf_s = self._lpf_scan(frames, bins)
        entropy, dsd_detected = self._spectral_entropy(frames), self._dsd_scan(frames, bins)

        side_anomaly = 0.0
        if side is not None:
            side_anomaly = self._side_channel_anomaly(frames_all, side, bins)

        lossy_score, lossy_ev, natural_score, natural_ev = self._score(cutoff_hz, cutoff_var, sharpness, cliff_depth, hf_ratio, nf_above, banding, side_anomaly, entropy, dsd_detected)
        net_score = max(0, lossy_score - natural_score)

        # ------------------------------------------------------------------
        # Advanced 11-rule forensic suite → unified Main Score (0–100)
        # ------------------------------------------------------------------
        main = round(net_score * 45 / self.MAX_LOSSY_SCORE)
        cassette_detected, vinyl_detected = False, False
        mp3_profile_match = any(abs(cutoff_hz - freq) <= 300 for freq in self.MP3_CUTOFFS.values()) and cutoff_hz < 20000

        if _SCIPY_OK:
            # Rule: Fakin' the Funk header integrity
            decoded_dur = len(audio) / self.sample_rate if max_seconds is None else 0.0
            dur_mm, br_mm, hdr_reasons = self._check_header_integrity(decoded_dur)
            result.header_duration_mismatch, result.header_bitrate_mismatch = dur_mm, br_mm
            if dur_mm: main += 20
            if br_mm: main += 25
            lossy_ev.extend(hdr_reasons)

            # Rule: psychoacoustic artefacts (pre-echo / aliasing / MP3 subband comb)
            psy_score, psy_ev, preecho, aliasing, mp3_noise = self._psychoacoustic_artifacts(audio, frames, bins, cutoff_hz, mp3_profile_match)
            result.preecho_pct, result.aliasing_corr, result.mp3_noise_pattern_detected = preecho, aliasing, mp3_noise
            main += psy_score
            lossy_ev.extend(psy_ev)

            # Rule 11: cassette source profiler (veto — analog tape, not codec damage)
            cass_score, cass_ev = self._cassette_source(audio, cutoff_hz, cutoff_std, mp3_noise)
            result.cassette_score = cass_score
            cassette_detected = cass_score >= 30
            if cassette_detected:
                main -= 40
                natural_ev.append(f"Cassette Source Profile matched (score {cass_score}): low cutoff is analog tape physics, not a codec wall.")
                natural_ev.extend(cass_ev)

            # Shared noise-floor measurement above the cutoff (FFT brickwall, computed once).
            # Band needs >=400 Hz of room; cutoffs above 93% of Nyquist are mastering-SRC
            # territory and excluded from void forensics.
            noise_band, void_db = None, 0.0
            band_lo, band_hi = cutoff_hz + 800.0, self.nyquist - 100.0
            if 0 < cutoff_hz < self.nyquist * 0.93 and band_hi - band_lo >= 400.0:
                cap = audio[: int(self.TIME_DOMAIN_CAP_S * self.sample_rate)]
                noise_band = self._fft_band_extract(cap, band_lo, band_hi)
                void_db = 20 * math.log10(float(np.sqrt(np.mean(noise_band ** 2))) + 1e-12)

            # Rule: AFD PRO segment voting (skipped under cassette veto).
            # Adaptive wall: a steep cliff (>30 dB) with a verified digital void above it
            # IS a codec wall wherever it sits — track it instead of the fixed 16.5 kHz so
            # high-cutoff encoders (LAME 320 walls at ~20.2 kHz) cannot slip past the vote.
            # Natural fades have shallow cliffs and dark analog sources leave hiss, so
            # neither can arm this.
            wall_hz = 16500.0
            if noise_band is not None and void_db < -85.0 and cliff_depth > 30.0:
                wall_hz = max(wall_hz, cutoff_hz + 400.0)
            seg_walled, seg_total, seg_fail = self._segment_voting(audio, wall_hz=wall_hz)
            result.segment_walled, result.segment_total, result.segment_wall_hz = seg_walled, seg_total, wall_hz
            if seg_fail and not cassette_detected:
                main += 55
                lossy_ev.append(f"Segment Vote FAILED: {seg_walled}/{seg_total} sampled 2s clips are frequency-walled at ≤{wall_hz / 1000:.1f} kHz — consistent whole-file lossy ancestry.")

            # Rule: silence dither / vinyl noise / clicks (3-phase)
            if cutoff_hz <= 21500 and not cassette_detected:
                sil_score, sil_ev, sil_ratio, vinyl_detected, clicks = self._silence_and_vinyl(audio, cutoff_hz, noise_band=noise_band)
                result.silence_ratio, result.vinyl_noise_detected, result.vinyl_clicks_per_min = sil_ratio, vinyl_detected, clicks
                main += sil_score
                (lossy_ev if sil_score > 0 else natural_ev).extend(sil_ev)

            # Rule: auCDtect statistical bound frequency & high-band phase entropy
            auc_avg, auc_prob, auc_phase = self._aucdtect_features(frames, phase_act, bins)
            result.auc_avg_bound_freq, result.auc_prob_bound_freq, result.auc_phase_entropy = auc_avg, auc_prob, auc_phase
            if self.sample_rate >= 40000 and 0 < auc_avg < 16500 and not cassette_detected and not vinyl_detected:
                main += 25
                lossy_ev.append(f"auCDtect Bound Collapse: spectral scatter dies at {auc_avg:,.0f} Hz on average — the statistical void of a lossy codec.")
            if auc_phase > 4.5 and cutoff_hz < self.nyquist * 0.85:
                main += 10
                lossy_ev.append(f"High-Band Phase Disruption: phase-difference entropy {auc_phase:.2f} bits with a depressed cutoff — quantized HF phase relationships.")

            # Rule: spectral sparsity (psychoacoustic bin-zeroing below the cutoff)
            sparsity = self._spectral_sparsity(frames, bins, cutoff_hz)
            result.spectral_sparsity = sparsity
            if sparsity > 0.30 and cutoff_hz < self.nyquist * 0.95:
                main += 10
                lossy_ev.append(f"Psychoacoustic Holes: {sparsity * 100:.0f}% of bins below the cutoff are zeroed (<-95 dB) — codec bit-allocation footprint.")

            # Rule: ultrasonic envelope correlation (anti-forensic noise-injection exposure)
            ultra = self._ultrasonic_envelope_correlation(frames, bins)
            result.hf_envelope_correlation = ultra
            if ultra < 0.15 and 0 < auc_avg < cutoff_hz - 2000 and cutoff_hz > 16500:
                main += 15
                lossy_ev.append(f"Fake HF Noise Injection: ultrasonic band is statistically independent of the music (corr {ultra:.2f}) while organic scatter dies at {auc_avg:,.0f} Hz — noise pasted above a codec wall.")

        main = max(0, min(100, main))
        label, sentence, caveats = self._verdict(main, net_score, cutoff_hz, dsd_detected, cassette_detected, vinyl_detected)

        legit_cutoff = cutoff_hz > (self.nyquist * 0.85)

        result.cutoff_hz, result.cutoff_hz_str = cutoff_hz, f"{int(cutoff_hz):,} Hz"
        result.cutoff_variance, result.cutoff_variance_interp = cutoff_var, self._interp_variance(cutoff_var, legit_cutoff)
        result.cutoff_sharpness_db, result.cutoff_sharpness_interp = sharpness, self._interp_sharpness(sharpness)
        result.cliff_depth_db = cliff_depth
        result.hf_energy_ratio, result.hf_energy_interp = hf_ratio, self._interp_hf_ratio(hf_ratio)
        result.banding_score, result.banding_interp = banding, self._interp_banding(banding)
        result.nf_above_cutoff_db, result.nf_interp = nf_above, self._interp_nf(nf_above)
        result.side_anomaly_score, result.side_interp = side_anomaly, self._interp_side(side_anomaly)
        result.entropy, result.entropy_interp = entropy, self._interp_entropy(entropy, legit_cutoff)
        result.lpf_detected, result.lpf_cutoff_str, result.dsd_detected = lpf_detected, lpf_s, dsd_detected
        result.lossy_score, result.natural_score, result.net_score, result.max_score = lossy_score, natural_score, net_score, self.MAX_LOSSY_SCORE
        result.raw_lossy_pct = min(100.0, lossy_score / self.MAX_LOSSY_SCORE * 100.0) if lossy_score > 0 else 0.0
        result.main_score = main
        result.net_confidence_pct = float(main)
        result.sparsity_interp = self._interp_sparsity(result.spectral_sparsity, legit_cutoff)
        result.hf_env_corr_interp = self._interp_ultra_corr(result.hf_envelope_correlation)
        result.auc_bound_interp = self._interp_bound(result.auc_avg_bound_freq)
        result.auc_phase_interp = self._interp_phase_entropy(result.auc_phase_entropy, legit_cutoff)
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
    try: claimed_bitrate = int(re.sub(r"[^\d]", "", tech.bit_rate) or 0)
    except ValueError: claimed_bitrate = 0

    auth = AuthenticityReport()
    engine = SpectralEngine(filepath, sample_rate, channels=channels,
                            claimed_duration=tech.duration_sec, claimed_bitrate_kbps=claimed_bitrate)
    auth.spectral = engine.analyse(max_seconds=fast_secs)

    auth.spectral_cutoff_hz = auth.spectral.cutoff_hz_str
    auth.spectral_cutoff_verdict = auth.spectral.primary_verdict
    auth.lpf_detected = auth.spectral.lpf_detected
    auth.lpf_cutoff_hz = auth.spectral.lpf_cutoff_str
    auth.cassette_rip_detected = auth.spectral.cassette_score >= 30
    auth.vinyl_rip_detected = auth.spectral.vinyl_noise_detected
    auth.side_channel_analysis = f"{auth.spectral.side_anomaly_score:.3f} {auth.spectral.side_interp}" if channels >= 2 else "mono — no side channel"
    if auth.spectral.header_duration_mismatch or auth.spectral.header_bitrate_mismatch:
        kinds = [k for k, f in (("duration", auth.spectral.header_duration_mismatch), ("bitrate", auth.spectral.header_bitrate_mismatch)) if f]
        auth.header_integrity = f"⚠ Header {' & '.join(kinds)} mismatch — forged or truncated stream"
    elif auth.spectral.scipy_available and auth.spectral.verdict_label != "INCONCLUSIVE":
        auth.header_integrity = "✓ Container header matches decoded stream"
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

def _main_score_colour(score: int) -> str:
    if score >= 86: return C.RED
    if score >= 55: return C.ORANGE
    if score >= 31: return C.YELLOW
    return C.GREEN

def _bound_colour(hz: float, nyquist_hint: float = 22050.0) -> str:
    if hz <= 0: return C.WHITE
    if hz >= min(18500.0, nyquist_hint * 0.85): return C.GREEN
    if hz >= 16500: return C.YELLOW
    return C.RED

def _phase_ent_colour(e: float) -> str:
    if e <= 0: return C.WHITE
    return C.GREEN if e < 4.5 else C.YELLOW

def _sparsity_colour(s: float) -> str:
    if s < 0.05: return C.GREEN
    if s < 0.30: return C.WHITE
    return C.ORANGE

def _ultra_corr_colour(c: float) -> str:
    if c > 0.3: return C.GREEN
    if c > 0.15: return C.WHITE
    return C.YELLOW

def _preecho_colour(p: float) -> str:
    if p < 5: return C.GREEN
    if p <= 10: return C.YELLOW
    return C.ORANGE

def _aliasing_colour(a: float) -> str:
    if a < 0.3: return C.GREEN
    if a <= 0.5: return C.YELLOW
    return C.ORANGE

def _silence_ratio_colour(r: float) -> str:
    if r < 0: return C.GREY
    if r < 0.15: return C.GREEN
    if r <= 0.3: return C.YELLOW
    return C.RED

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
        print(f"  {_c(_main_score_colour(sp.main_score), f'Main Score: {sp.main_score}/100')}  {_c(C.GREY, '(0 = pristine lossless · 100 = certain transcode)')}")
        print(f"  {_c(C.GREY, f'Base engine: Lossy {sp.lossy_score} − Natural {sp.natural_score} = Net {sp.net_score}/{sp.max_score}  |  Raw Error Rate: {sp.raw_lossy_pct:.1f}%')}")
        print()
        rows_spec = [
            _kv("Ultrasonic Noise", _c(C.ORANGE, "⚠ DSD/SACD Transcode Profile") if sp.dsd_detected else _c(C.GREEN, "✓ Normal")),
            _kv("HF Cutoff",         sp.cutoff_hz_str),
            _kv("Cutoff Variance",   f"{sp.cutoff_variance:.1f} Hz²  " + _c(C.GREY, sp.cutoff_variance_interp)),
            _kv("Cliff Sharpness",   f"{sp.cutoff_sharpness_db:.1f} dB/bin · {sp.cliff_depth_db:.0f} dB drop/800Hz  " + _c(C.GREY, sp.cutoff_sharpness_interp)),
            _kv("HF Energy Ratio",   f"{sp.hf_energy_ratio:.5f}  " + _c(C.GREY, sp.hf_energy_interp)),
            _kv("Side Anomaly",      f"{sp.side_anomaly_score:.3f}  " + _c(C.GREY, sp.side_interp)),
            _kv("Banding Score",     f"{sp.banding_score:.3f}  " + _c(C.GREY, sp.banding_interp)),
            _kv("NF Above Cutoff",   f"{sp.nf_above_cutoff_db:.1f} dB  " + _c(C.GREY, sp.nf_interp)),
            _kv("LPF",              ("⚠ YES — " + sp.lpf_cutoff_str) if sp.lpf_detected else "✓ None detected"),
            _kv("Spectral Entropy", f"{sp.entropy:.3f}  " + _c(C.GREY, sp.entropy_interp)),
        ]
        for row in rows_spec:
            if row: print(row)

        print(_subsection("Advanced DSP Forensics  (scipy suite)"))
        if sp.scipy_available:
            if sp.segment_walled < 0: seg_val = _c(C.GREY, "n/a — file too short for 7-segment voting")
            else:
                seg_majority = sp.segment_walled > sp.segment_total / 2
                seg_col = C.RED if seg_majority else (C.YELLOW if sp.segment_walled > 0 else C.GREEN)
                seg_state = "✗ FAILED" if seg_majority else ("~ partial walls" if sp.segment_walled > 0 else "✓ passed")
                seg_val = _c(seg_col, f"{seg_state} — {sp.segment_walled}/{sp.segment_total} clips walled ≤{sp.segment_wall_hz / 1000:.1f} kHz")
            if sp.silence_ratio < 0: sil_val = _c(C.GREY, "n/a — no silent passages ≥ 0.5s found")
            else: sil_val = _c(_silence_ratio_colour(sp.silence_ratio), f"{sp.silence_ratio:.3f}") + "  " + _c(C.GREY, "[<0.15 clean · >0.3 codec hash in silence]")
            if sp.vinyl_noise_detected: vinyl_val = _c(C.BLUE, f"✓ surface noise detected ({sp.vinyl_clicks_per_min:.0f} clicks/min)")
            else: vinyl_val = _c(C.GREY, "not detected")
            if sp.cassette_score >= 30: cass_val = _c(C.BLUE, f"✓ tape profile matched (score {sp.cassette_score}/80)")
            elif sp.cassette_score > 0: cass_val = _c(C.GREY, f"weak match (score {sp.cassette_score}/80)")
            else: cass_val = _c(C.GREY, "not detected")
            hdr_mm = sp.header_duration_mismatch or sp.header_bitrate_mismatch
            rows_adv = [
                _kv("Header Integrity", _c(C.RED, "⚠ header/stream mismatch — forged or truncated") if hdr_mm else _c(C.GREEN, "✓ container matches decoded stream")),
                _kv("Segment Vote", seg_val),
                _kv("auCDtect Bound", _c(_bound_colour(sp.auc_avg_bound_freq), f"{sp.auc_avg_bound_freq:,.0f} Hz avg · {sp.auc_prob_bound_freq:,.0f} Hz mode") + "  " + _c(C.GREY, sp.auc_bound_interp)),
                _kv("HF Phase Entropy", _c(_phase_ent_colour(sp.auc_phase_entropy), f"{sp.auc_phase_entropy:.2f} bits") + "  " + _c(C.GREY, sp.auc_phase_interp)),
                _kv("Spectral Sparsity", _c(_sparsity_colour(sp.spectral_sparsity), f"{sp.spectral_sparsity:.3f}") + "  " + _c(C.GREY, sp.sparsity_interp)),
                _kv("Ultrasonic Corr.", _c(_ultra_corr_colour(sp.hf_envelope_correlation), f"{sp.hf_envelope_correlation:+.2f}") + "  " + _c(C.GREY, sp.hf_env_corr_interp)),
                _kv("Pre-Echo", _c(_preecho_colour(sp.preecho_pct), f"{sp.preecho_pct:.1f}% of transients") + "  " + _c(C.GREY, "[MDCT block smearing]")),
                _kv("HF Aliasing Corr.", _c(_aliasing_colour(sp.aliasing_corr), f"{sp.aliasing_corr:.2f}") + "  " + _c(C.GREY, "[codec filterbank mirroring]")),
                _kv("MP3 Subband Comb", _c(C.ORANGE, "⚠ 689 Hz comb structure detected") if sp.mp3_noise_pattern_detected else _c(C.GREEN, "✓ none")),
                _kv("Silence Dither", sil_val),
                _kv("Vinyl Source", vinyl_val),
                _kv("Cassette Source", cass_val),
            ]
            for row in rows_adv:
                if row: print(row)
        else:
            print(f"  {_c(C.YELLOW, '⚠ scipy not installed — advanced forensic suite skipped (pip install scipy)')}")

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
    source_flags = []
    if auth.cassette_rip_detected: source_flags.append("cassette tape")
    if auth.vinyl_rip_detected: source_flags.append("vinyl")
    for row in [_kv("Bit-Depth Auth", auth.bit_depth_authentic), _kv("Header Integrity", auth.header_integrity), _kv("Analog Source", _c(C.BLUE, " + ".join(source_flags) + " signature detected") if source_flags else ""), _kv("Side Channel", auth.side_channel_analysis), _kv("Phase Corr.", f"{auth.phase_correlation} {auth.phase_verdict}" if auth.phase_correlation else ""), _kv("Clipping", auth.clipping_verdict if auth.clipping_verdict else ""), _kv("Silence", auth.silence_total_pct)]:
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
