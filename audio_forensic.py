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
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    from scipy.fft import rfft as _srfft, irfft as _sirfft, rfftfreq as _srfftfreq, next_fast_len as _next_fast_len
    from scipy.ndimage import uniform_filter1d as _uniform1d
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

# ---------------------------------------------------------------------------
# Live progress (single status line on stderr; thread-safe; TTY only)
# ---------------------------------------------------------------------------
class _Status:
    enabled = sys.stderr.isatty()
    _lock = threading.Lock()
    _active: "dict[str, tuple[str, float]]" = {}
    _done = 0
    _total = 0
    _file_times: "list[float]" = []

    # Cumulative progress fraction at the START of each stage (profiled on the 4-min
    # reference, fake path — the worst case). ETA extrapolates elapsed/(progress) so
    # the estimate self-calibrates to the machine; no absolute speed model needed.
    _STAGE_PROGRESS = {
        "probing metadata": 0.0, "decoding": 0.02, "STFT": 0.14, "spectral metrics": 0.24,
        "resample check": 0.28,
        "header integrity": 0.29, "psychoacoustic tests": 0.30, "cassette profile": 0.41,
        "segment voting": 0.55, "silence & vinyl analysis": 0.57, "auCDtect statistics": 0.63,
        "waiting on loudness/spectrogram": 0.80, "finalizing": 0.97,
    }

    _workers = 1

    @classmethod
    def begin(cls, total: int, workers: int = 1) -> None:
        cls._total, cls._done, cls._active, cls._file_times = total, 0, {}, []
        cls._workers = max(1, workers)

    @classmethod
    def update(cls, name: str, stage: str) -> None:
        if not cls.enabled: return
        with cls._lock:
            started = cls._active.get(name, ("", time.perf_counter()))[1]
            cls._active[name] = (stage, started)
            cls._render()

    @classmethod
    def done(cls, name: str) -> None:
        if not cls.enabled: return
        with cls._lock:
            entry = cls._active.pop(name, None)
            if entry is not None:
                cls._file_times.append(time.perf_counter() - entry[1])
            cls._done += 1
            cls._render()

    @classmethod
    def clear(cls) -> None:
        if not cls.enabled: return
        with cls._lock:
            sys.stderr.write("\r\x1b[2K"); sys.stderr.flush()

    @classmethod
    def _render(cls) -> None:
        now = time.perf_counter()
        parts, active_etas = [], []
        for n, (s, t0) in cls._active.items():
            elapsed = now - t0
            p = cls._STAGE_PROGRESS.get(s, 0.0)
            if p >= 0.05:
                eta = elapsed * (1.0 - p) / p
                active_etas.append(eta)
                bar = "▰" * int(p * 6) + "▱" * (6 - int(p * 6))
                parts.append(f"{n[:24]}: {s} {bar} ~{max(0.0, eta):.0f}s")
            else:
                parts.append(f"{n[:24]}: {s} ({elapsed:.0f}s)")
        line = f"⏳ [{cls._done}/{cls._total}] " + "  ·  ".join(parts)
        # Batch ETA: slowest active file + queued files spread across the worker pool
        queued = cls._total - cls._done - len(cls._active)
        if cls._total > 1 and active_etas and (queued == 0 or cls._file_times):
            total_eta = max(active_etas)
            if queued > 0:
                total_eta += queued * (sum(cls._file_times) / len(cls._file_times)) / cls._workers
            line += f"  ·  batch ~{total_eta:.0f}s left"
        width = shutil.get_terminal_size((120, 20)).columns - 1
        sys.stderr.write("\r\x1b[2K" + line[:width]); sys.stderr.flush()

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
    other: dict[str, str] = field(default_factory=dict)   # every remaining mediainfo tag

@dataclass
class AudioTechnical:
    bit_rate: str = ""; channels: str = ""; precision: str = ""; sample_rate: str = ""
    sample_encoding: str = ""; duration: str = ""; duration_sec: float = 0.0
    writing_library: str = ""; format_profile: str = ""; compression_mode: str = ""
    codec: str = ""   # raw mediainfo Format of the audio stream (container-independent)

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
    segment_map: list[str] = field(default_factory=list)
    codec_fingerprint: str = ""
    resample_detected: str = ""; resample_src_rate: int = 0
    fake_hires: str = ""
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
    mqa_detected: bool = False
    side_channel_analysis: str = ""
    header_integrity: str = ""
    encoder_trace: str = ""

@dataclass
class ForensicReport:
    filepath: Path; tags: AudioTags = field(default_factory=AudioTags)
    technical: AudioTechnical = field(default_factory=AudioTechnical)
    sox_stats: dict[str, str] = field(default_factory=dict)
    loudness: LoudnessProfile = field(default_factory=LoudnessProfile)
    authenticity: AuthenticityReport = field(default_factory=AuthenticityReport)
    dr_score: str = "N/A"; spectrogram_path: Optional[Path] = None
    analysis_seconds: float = 0.0
    @property
    def file_size_mb(self) -> float: return self.filepath.stat().st_size / (1024 * 1024)

# ---------------------------------------------------------------------------
# Tool Extractors
# ---------------------------------------------------------------------------
# General-track keys that are technical/duplicated elsewhere in the report — everything
# NOT in this set (and not a known tag) flows into tags.other so no metadata is hidden.
_MEDIAINFO_NONTAG_KEYS = {
    "@type", "AudioCount", "VideoCount", "ImageCount", "MenuCount", "TextCount",
    "FileExtension", "FileSize", "Duration", "OverallBitRate", "OverallBitRate_Mode",
    "StreamSize", "IsStreamable", "FrameRate", "FrameCount", "HeaderSize", "DataSize",
    "FooterSize", "CompleteName", "FileName", "FileNameExtension", "FolderName",
    "File_Created_Date", "File_Created_Date_Local", "File_Modified_Date", "File_Modified_Date_Local",
    "Audio_Format_List", "Audio_Format_WithHint_List", "Audio_Codec_List", "Audio_Language_List",
    "Format", "Format_Profile", "Format_Version", "Cover_Data",
}
_KNOWN_TAG_KEYS = {
    "Title", "Album", "Recorded_Date", "Album_Performer", "Album_Artist", "Performer",
    "BPM", "Comment", "REPLAYGAIN_TRACK_GAIN", "REPLAYGAIN_ALBUM_GAIN",
}

def _prettify_mi_key(key: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", key.replace("_", " ")).strip().title()

def extract_mediainfo(filepath: Path) -> tuple[AudioTags, AudioTechnical]:
    result = _run(["mediainfo", "--Output=JSON", str(filepath)])
    if result.returncode != 0: return AudioTags(), AudioTechnical()

    data = json.loads(result.stdout)
    tags, tech = AudioTags(), AudioTechnical()

    for track in data.get("media", {}).get("track", []):
        t = track.get("@type")
        if t == "General":
            extra = track.get("extra", {}) or {}
            tags.title = track.get("Title", track.get("Track", ""))
            tags.album = track.get("Album", "")
            tags.date = track.get("Recorded_Date", "")
            tags.album_artist = track.get("Album_Performer", track.get("Album_Artist", ""))
            tags.artist = track.get("Performer", "")
            tags.bpm = track.get("BPM", "")
            tags.comments = track.get("Comment", extra.get("Comment", ""))
            tags.comment_quality = extra.get("commentQuality", "")
            tags.replaygain_track_gain = extra.get("REPLAYGAIN_TRACK_GAIN", track.get("REPLAYGAIN_TRACK_GAIN", ""))
            tags.replaygain_album_gain = extra.get("REPLAYGAIN_ALBUM_GAIN", track.get("REPLAYGAIN_ALBUM_GAIN", ""))
            # Philosophy: surface EVERY remaining tag the file carries.
            merged = {**track, **extra}
            for key, val in merged.items():
                if key in _MEDIAINFO_NONTAG_KEYS or key in _KNOWN_TAG_KEYS or key == "extra": continue
                if not isinstance(val, str) or not val.strip(): continue
                if key.lower().startswith("replaygain"): continue
                if len(val) > 200: val = val[:200] + " …"
                tags.other[_prettify_mi_key(key)] = val
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
            tech.codec = fmt
            tech.sample_encoding = f"{bit_depth}-bit {fmt}" if bit_depth else fmt
            tech.writing_library = track.get("Encoded_Library__String", track.get("Encoded_Library", ""))
            tech.format_profile = track.get("Format_Profile", "")
            tech.compression_mode = track.get("Compression_Mode", "")
            mins, secs = divmod(int(raw_dur), 60)
            tech.duration = f"{mins:02d}:{secs:02d}"

    return tags, tech

_LOSSLESS_EXTS = {".flac", ".wav", ".alac", ".m4a", ".ape", ".wv", ".aiff", ".aif"}
_ENCODER_SIGNATURES = ("lame", "libmp3lame", "fraunhofer", " fhg", "nero aac", "fdk-aac",
                       "320kbps", "320 kbps", "v0 (vbr", "joint stereo", "xing")

def detect_encoder_trace(tags: AudioTags, tech: AudioTechnical, filepath: Path) -> str:
    """Lossy-encoder fingerprints left in a lossless container's metadata are a
    transcode confession the spectrum can't even see. Display-level red flag only
    (tags can be innocent quotes), not scored."""
    if filepath.suffix.lower() not in _LOSSLESS_EXTS: return ""
    hay = " ".join([tags.comments, tags.comment_quality, tech.writing_library,
                    *tags.other.values()]).lower().replace("mp3tag", "")  # the tagger app is innocent
    hits = sorted({sig.strip() for sig in _ENCODER_SIGNATURES if sig in hay})
    if not hits: return ""
    return f"⚠ Lossy encoder fingerprint in metadata: {', '.join(hits)} — tags survived a transcode"

_MQA_MAGIC = 0xbe0498c88  # 36-bit MQA sync word (reverse-engineered, MQA_identifier project)

def detect_mqa(tags: AudioTags, tech: AudioTechnical, filepath: Path) -> str:
    """MQA folds 'hi-res' data into the LSBs as pseudo-noise dither — spectrally
    invisible to PCM forensics (the stream verifies as ordinary 16/44.1 lossless).
    Two layers: metadata traces (fast), then the signal itself — MQA carries a
    control stream in (L XOR R) at bit position (depth−16) that begins with a
    36-bit sync word. Tag-stripping can't remove that."""
    hay = " ".join([tech.writing_library, tech.format_profile, tags.comments,
                    *tags.other.keys(), *tags.other.values()]).lower()
    found = "metadata tags" if "mqa" in hay else ""

    depth = 0
    try: depth = int(tech.precision.replace("-bit", "").strip())
    except ValueError: pass
    if not found and _NUMPY_OK and depth in (16, 24) and tech.channels.strip() == "2":
        # First 4 s, bit-exact stereo decode (s32le: sample sits in the top bits)
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(filepath), "-vn", "-t", "4",
                            "-ac", "2", "-c:a", "pcm_s32le", "-f", "s32le", "pipe:1"],
                           capture_output=True, check=False)
        if r.returncode == 0 and len(r.stdout) >= 8 * 4096:
            arr = np.frombuffer(r.stdout[: len(r.stdout) // 8 * 8], dtype=np.int32).reshape(-1, 2)
            x = np.bitwise_xor(arr[:, 0], arr[:, 1]).view(np.uint32) >> np.uint32(32 - depth)
            weights = 2.0 ** np.arange(35, -1, -1)  # rolling 36-bit window, MSB first
            for pos in (depth - 16, depth - 15, depth - 14):
                bits = ((x >> np.uint32(pos)) & 1).astype(np.float64)
                if len(bits) < 36: break
                vals = np.lib.stride_tricks.sliding_window_view(bits, 36) @ weights
                if np.any(vals == float(_MQA_MAGIC)):
                    found = f"sync word in the bitstream (bit {pos})"
                    break
    if not found: return ""
    return (f"MQA-encoded stream (detected via {found}) — high-frequency content is "
            "origami-folded into the low-order bits as pseudo-noise. Spectral forensics see "
            "only the PCM core; the folded payload (and its lossy unfold) cannot be verified.")

_SOX_UNSUPPORTED = {".m4a", ".mp4", ".aac", ".ogg", ".opus", ".wma", ".ape", ".mp3", ".dff", ".dsf"}

def extract_sox_stats(filepath: Path) -> dict[str, str]:
    if filepath.suffix.lower() in _SOX_UNSUPPORTED:
        # SoX can't read these natively — pipe a WAV decode straight from ffmpeg
        # into SoX's stdin (no temp file, stays in RAM).
        decode = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(filepath), "-vn",
             "-ac", "2", "-sample_fmt", "s16", "-f", "wav", "pipe:1"],
            capture_output=True, check=False)
        if decode.returncode != 0 or not decode.stdout: return {}
        result = subprocess.run(["sox", "-t", "wav", "-", "-n", "stat"],
                                input=decode.stdout, capture_output=True, check=False)
        stderr_text = result.stderr.decode("utf-8", errors="replace")
    else:
        stderr_text = _run(["sox", str(filepath), "-n", "stat"]).stderr

    stats: dict[str, str] = {}
    for line in stderr_text.splitlines():
        if ":" not in line: continue
        raw_key, _, raw_val = line.partition(":")
        if key := _camel_case(raw_key.strip()): stats[key] = raw_val.strip()
    return stats

def extract_loudness(filepath: Path) -> tuple[LoudnessProfile, str]:
    """Single ffmpeg invocation: the stream is decoded once and split through
    astats, ebur128 and drmeter simultaneously. Returns (profile, DR score)."""
    graph = ("[0:a]asplit=3[a1][a2][a3];"
             "[a1]astats[o1];"
             "[a2]aresample=48000,ebur128=peak=true[o2];"
             "[a3]drmeter[o3]")
    r = _run(["ffmpeg", "-i", str(filepath), "-vn", "-filter_complex", graph,
              "-map", "[o1]", "-map", "[o2]", "-map", "[o3]", "-f", "null", "-"])

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

    def _field(pat: str) -> str:
        matches = re.findall(pat, r.stderr)
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

    dr_match = re.search(r"DR:\s+([\d.]+)", r.stderr)
    dr_score = f"DR{int(float(dr_match.group(1)))}" if dr_match else "N/A"
    return lp, dr_score

# --- Bit-depth forensics constants -------------------------------------------
# A bit rank must fire in >= this fraction of samples to count as "in use"
# (robust to stray corrupt samples / a handful of denormal values).
_BD_ACTIVE_FRAC = 1e-4
# TPDF-dithered 16-bit noise floor: 6.02*N - 3 dB below full scale. Sample-rate
# independent (the floor LEVEL doesn't move with SR — it just spreads over a wider
# band). Sources: audiocheck.net/audiotests_dithering, tonmeister "High-Res Part 6".
_BD_16BIT_FLOOR_DBFS = -93.0
# To read a noise floor at all, the quietest sustained window must drop below this.
# Loudness-war commercial masters sit far above it (measured quietest-window RMS:
# -26 dBFS web hi-res, -34 redbook, -59 MFSL) -> the floor is masked -> we ABSTAIN.
_BD_FLOOR_EXPOSED_DBFS = -86.0
# A floor below this proves content exists beneath the 16-bit dither floor -> the
# source genuinely carries >16-bit information (cannot be a 16-bit upsample). Set a
# few dB under the dither floor's lower edge (TPDF runs -93..-99 dBFS depending on
# the dither) so a real 16-bit floor never tips into a "genuine hi-res" reading.
_BD_GENUINE_HIRES_DBFS = -102.0
# Only assert "sitting at the 16-bit dither level" when the flat floor is actually
# down near -93 dBFS; a flat floor higher than this is worse than 16-bit (limited DR).
_BD_16BIT_LEVEL_MAX_DBFS = -89.0
# |HF-LF| of the exposed floor below this => spectrally flat/white = TPDF dither
# signature (16-bit tell). Above it and LF-heavy => analog hiss (genuine analog).
_BD_FLOOR_FLAT_TOL_DB = 7.0


def _effective_bits(arr_i32: "np.ndarray", channels: int) -> int:
    """Highest bit-depth actually exercised, MSB-aligned, across all channels.

    ffmpeg renders an N-bit sample MSB-aligned in the 32-bit container (a 16-bit
    value lands on multiples of 2**16, a 24-bit value on multiples of 2**8). The
    lowest bit rank hit by >= _BD_ACTIVE_FRAC of a channel's nonzero samples marks
    that channel's live depth; we take the deepest channel so a one-channel pad (or
    a silent channel) can't mask bits the file really uses.
    """
    if channels < 1:
        channels = 1
    usable = (arr_i32.size // channels) * channels
    deck = arr_i32[:usable].reshape(-1, channels)
    best = 0
    for ch in range(channels):
        nz = deck[:, ch][deck[:, ch] != 0].astype(np.int64)
        if nz.size < 500:
            continue
        tz = np.log2((nz & -nz).astype(np.float64)).astype(np.int64)   # exact trailing zeros
        counts = np.bincount(tz, minlength=33)
        threshold = max(8, int(nz.size * _BD_ACTIVE_FRAC))
        cum = np.cumsum(counts)
        eff = 32 - int(np.argmax(cum >= threshold))
        best = max(best, eff)
    return best


def _noise_floor_profile(arr_i32: "np.ndarray", channels: int, sample_rate: int) -> "dict | None":
    """Expose the noise floor underneath the music, if the master has any quiet.

    Returns a dict {floor_db, peak_db, flat, slope_db} or None when the window is
    unusable. ``floor_db`` is the 1.5th-percentile broadband RMS over 100 ms blocks
    (full scale = 1.0); ``flat`` flags a white/TPDF-shaped floor (the 16-bit dither
    tell) vs. an LF-heavy analog floor; ``slope_db`` = HF-band minus LF-band level
    of the quietest blocks.
    """
    if not _NUMPY_OK or sample_rate < 8000:
        return None
    usable = (arr_i32.size // max(1, channels)) * max(1, channels)
    if usable < sample_rate:
        return None
    mono = arr_i32[:usable].reshape(-1, max(1, channels)).astype(np.float64).mean(axis=1) / (2.0 ** 31)
    block = sample_rate // 10
    n = len(mono) // block
    if n < 20:
        return None
    blocks = mono[: n * block].reshape(n, block)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    rms = rms[rms > 0]
    if rms.size < 20:
        return None
    floor_lin = float(np.percentile(rms, 1.5))
    peak_lin = float(np.percentile(rms, 99))
    if floor_lin <= 0:
        return None
    floor_db = 20.0 * math.log10(floor_lin)
    peak_db = 20.0 * math.log10(peak_lin) if peak_lin > 0 else 0.0

    # Spectral colour of the quietest 10% of blocks (noise-floor dominated).
    order = np.argsort(rms)
    quiet_idx = order[: max(3, n // 10)]
    quiet = blocks[quiet_idx] * np.hanning(block)
    spec = np.mean(np.abs(np.fft.rfft(quiet, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(block, 1.0 / sample_rate)
    lf = spec[(freqs > 150) & (freqs < 2000)]
    hf = spec[(freqs > sample_rate * 0.33) & (freqs < sample_rate * 0.45)]
    slope_db = float("nan")
    flat = False
    if lf.size and hf.size and lf.mean() > 0 and hf.mean() > 0:
        slope_db = 10.0 * math.log10(hf.mean() / lf.mean())
        flat = abs(slope_db) < _BD_FLOOR_FLAT_TOL_DB
    return {"floor_db": floor_db, "peak_db": peak_db, "flat": flat, "slope_db": slope_db}


def check_bit_depth_authenticity(filepath: Path, claimed_depth: int, duration_sec: float = 0.0,
                                 sample_rate: int = 0, channels: int = 0) -> str:
    """Two-prong effective-bit-depth forensics.

    Decodes a 30 s window (from the middle of the track — intros/outros are often
    quiet or faded) as interleaved-stereo 32-bit PCM (never mono-downmixed: averaging
    L+R injects a half-LSB and corrupts the bit pattern) and runs two independent
    tests:

    Prong 1 — used bits. The lowest bit rank actually exercised reveals clean
    integer zero-padding: 16-bit content shifted into a 24-bit container leaves the
    bottom 8 bits dead in every sample. Per-channel, robust to stray samples. This
    is a *proof* of padding when it fires, but it is BLIND to dithered/float/lossy
    upscales, whose low bits go live (so "all bits used" must NOT be reported as a
    confident "verified 24-bit" — that overclaims).

    Prong 2 — noise floor / effective dynamic range. The only signal that sees
    through a dithered upscale, but bounded by physics: a 16-bit step is detectable
    only when the source genuinely holds content below the 16-bit dither floor
    (-93 dBFS). On loud masters with no exposed floor the prong ABSTAINS (the honest
    answer); when a quiet passage exists it either CONFIRMS genuine >16-bit content
    (floor < -99 dBFS) or flags an effective-16-bit ceiling (a flat/white floor
    sitting right at -93 dBFS under a >16-bit container).
    """
    if not claimed_depth: return ""
    if not _NUMPY_OK: return f"claimed {claimed_depth}-bit — numpy unavailable, not verified"

    cmd = ["ffmpeg", "-v", "error"]
    if duration_sec > 70:
        cmd += ["-ss", f"{max(0.0, duration_sec / 2 - 15):.2f}"]
    cmd += ["-i", str(filepath), "-vn", "-t", "30", "-c:a", "pcm_s32le", "-f", "s32le", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0 or len(result.stdout) < 4096:
        return f"claimed {claimed_depth}-bit — decode failed, not verified"

    arr = np.frombuffer(result.stdout[: len(result.stdout) // 4 * 4], dtype=np.int32)
    nz_total = int(np.count_nonzero(arr))
    if nz_total < 1000:
        return f"claimed {claimed_depth}-bit — sampled window is silent, not verified"

    # Caller passes the probed channel count; fall back to interleave detection
    # (both lanes populated under a stereo assumption => stereo, else mono).
    if channels < 1:
        channels = 2 if arr.size >= 2 and np.count_nonzero(arr[1::2]) > 0 else 1
    effective_bits = _effective_bits(arr, channels)

    if sample_rate < 8000:
        # Last-resort rate estimate from the byte count (a full 30 s clip). The floor
        # LEVEL is sample-rate independent, so this only affects block sizing.
        span = min(30.0, duration_sec) if duration_sec else 30.0
        est = (arr.size / max(1, channels)) / span if span > 0 else 44100
        sample_rate = min((44100, 48000, 88200, 96000, 176400, 192000, 22050),
                          key=lambda r: abs(r - est))
    prof = _noise_floor_profile(arr, channels, sample_rate)
    return _bit_depth_verdict(claimed_depth, effective_bits, prof)


def _bit_depth_verdict(claimed_depth: int, effective_bits: int, prof: "dict | None") -> str:
    """Pure verdict logic for the two prongs (separated for unit testing).

    ``effective_bits`` is the deepest bit rank exercised (Prong 1); ``prof`` is the
    noise-floor profile from ``_noise_floor_profile`` or None when unmeasurable.
    """
    # Prong 1: a clean integer pad (low bits hard-zero) is conclusive.
    if effective_bits and effective_bits <= claimed_depth - 8:
        return (f"⚠ Upscaled: {claimed_depth}-bit container but only {effective_bits} bits carry "
                f"signal — clean integer pad from a {effective_bits}-bit source")
    if effective_bits and effective_bits < claimed_depth:
        return (f"~ {effective_bits} of {claimed_depth} bits exercised — reduced-depth master, "
                f"bit-shifted gain, or fixed-point chain (not zero-padded)")

    # Prong 2: bits are fully live. "All bits used" alone does NOT prove the source
    # depth (a dithered/float upscale fills them too) — consult the noise floor.
    if prof is None:
        return f"✓ {claimed_depth}-bit container fully exercised — source depth not independently confirmable"
    floor, flat = prof["floor_db"], prof["flat"]

    # No exposed floor (loud master) -> abstain honestly.
    if floor > _BD_FLOOR_EXPOSED_DBFS:
        return (f"✓ {claimed_depth}-bit container fully exercised — noise floor masked by a loud "
                f"master ({floor:.0f} dBFS), source depth not independently confirmable")

    # Floor proves content below the 16-bit dither floor -> genuine high-res.
    if floor < _BD_GENUINE_HIRES_DBFS:
        eff_dr = max(claimed_depth, int(round((-floor - 1.76) / 6.02)))
        return (f"✓ Genuine {claimed_depth}-bit — noise floor at {floor:.0f} dBFS confirms content "
                f"below the 16-bit limit (~{eff_dr}-bit dynamic range)")

    # Floor is exposed but sits in 16-bit territory (-102 .. -86 dBFS).
    eff_dr = int(round((-floor - 1.76) / 6.02))
    if claimed_depth >= 24 and flat and floor <= _BD_16BIT_LEVEL_MAX_DBFS:
        # Flat/white floor right at the 16-bit dither level = the TPDF tell.
        return (f"⚠ Effective ~16-bit — {claimed_depth}-bit container but a flat noise floor at "
                f"{floor:.0f} dBFS (the 16-bit dither level) — upsampled from 16-bit")
    if claimed_depth >= 24:
        colour = "colored/analog" if not flat else "limited dynamic range"
        return (f"~ Noise floor {floor:.0f} dBFS (~{eff_dr}-bit effective, {colour}) — consistent with "
                f"an analog-sourced or heavily-compressed {claimed_depth}-bit master; source depth unconfirmable")
    # claimed 16-bit (or 20) with a floor at the 16-bit level == consistent.
    return f"✓ {claimed_depth}-bit consistent — noise floor at {floor:.0f} dBFS matches the claimed depth"

# --- Byproduct metrics: computed from the SpectralEngine's decoded audio.
#     Replaces three full ffmpeg invocations (aphasemeter, astats clipping,
#     silencedetect) — two of which were silently broken filter syntax anyway.

def measure_phase_correlation(mid: "np.ndarray | None", side: "np.ndarray | None", sample_rate: int) -> tuple[str, str]:
    """Mean per-100ms Pearson correlation between L and R (1 mono · 0 uncorrelated · -1 antiphase)."""
    if not _NUMPY_OK or mid is None or side is None: return "", ""
    left, right = mid + side, mid - side
    block = max(1, sample_rate // 10)
    n = len(left) // block
    if n < 1: return "", ""
    L = left[: n * block].reshape(n, block).astype(np.float64)
    R = right[: n * block].reshape(n, block).astype(np.float64)
    L -= L.mean(axis=1, keepdims=True); R -= R.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(L * L, axis=1) * np.sum(R * R, axis=1))
    valid = denom > 1e-12
    if not valid.any(): return "", ""
    avg = float(np.mean(np.sum(L * R, axis=1)[valid] / denom[valid]))
    if avg >= 0.9: return f"{avg:.3f}", "Mono-compatible"
    elif avg >= 0.5: return f"{avg:.3f}", "Normal stereo"
    elif avg >= 0.0: return f"{avg:.3f}", "Wide stereo"
    elif avg >= -0.3: return f"{avg:.3f}", "⚠ Possible fake stereo / heavy M-S processing"
    else: return f"{avg:.3f}", "⚠ Phase cancellation — check mono fold-down"

def detect_clipping(mid: "np.ndarray | None", side: "np.ndarray | None") -> tuple[str, str]:
    """Counts samples at digital full scale (≥ 16-bit ceiling) across both channels."""
    if not _NUMPY_OK or mid is None: return "", ""
    threshold = 1.0 - 1.0 / 32768
    if side is not None:
        total = int(np.sum(np.abs(mid + side) >= threshold) + np.sum(np.abs(mid - side) >= threshold))
    else:
        total = int(np.sum(np.abs(mid) >= threshold))
    if total == 0: return "0", "✓ No clipped samples"
    elif total < 10: return str(total), f"~ {total} clipped sample(s) — minor"
    else: return str(total), f"⚠ {total:,} clipped samples — audible distortion likely"

def _noise_floor_from_audio(mid: "np.ndarray | None", sample_rate: int) -> str:
    """Fallback noise floor: 5th percentile of per-100ms block RMS, in dBFS."""
    if not _NUMPY_OK or mid is None or len(mid) < sample_rate: return ""
    block = sample_rate // 10
    n = len(mid) // block
    rms = np.sqrt(np.mean(mid[: n * block].reshape(n, block).astype(np.float64) ** 2, axis=1))
    rms = rms[rms > 0]
    if rms.size < 5: return ""
    return f"{20 * math.log10(float(np.percentile(rms, 5))):.2f}"

def map_silence(mid: "np.ndarray | None", sample_rate: int, duration_sec: float) -> tuple[str, list[str]]:
    """Silent passages (< -60 dBFS for ≥ 0.5 s), vectorized run detection."""
    if not _NUMPY_OK or mid is None or len(mid) == 0: return "", []
    is_sil = np.abs(mid) < 10 ** (-60.0 / 20.0)
    padded = np.concatenate(([False], is_sil, [False]))
    d = np.diff(padded.astype(np.int8))
    starts_i, ends_i = np.where(d == 1)[0], np.where(d == -1)[0]
    min_samples = int(0.5 * sample_rate)
    segs = [(s / sample_rate, e / sample_rate) for s, e in zip(starts_i, ends_i) if (e - s) >= min_samples]
    span = duration_sec if duration_sec > 0 else len(mid) / sample_rate
    total_silent = sum(e - s for s, e in segs)
    pct = (total_silent / span * 100) if span > 0 else 0
    sections = []
    eof_cut = (len(mid) - 2) / sample_rate
    for s, e in segs:
        marker = " → EOF" if e >= eof_cut else ""
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


def generate_spectrogram(filepath: Path, duration_sec: float = 0.0) -> Optional[Path]:
    """Generates a clean mono spectrogram (SoX rendering — best visual quality).

    The decoded mono WAV is piped from ffmpeg straight into SoX's stdin: no temp
    file, no disk I/O. Height is 513 px (a power of two + 1) — SoX maps that to
    an efficient DFT size; 512 forces a pathological resampling path ~20x slower.
    Falls back to ffmpeg showspectrumpic if SoX fails.

    ffmpeg can't seek back on the pipe to patch the WAV data-chunk size, so SoX
    reads a bogus length ("Premature EOF on .wav input file") and can't scale the
    time axis to ``-x``, collapsing the image to a ~150 px barcode. Feeding the
    known container duration via the spectrogram ``-d`` effect restores the full
    1280 px width. NB: plain seconds only — a bare ``s`` suffix means *samples*
    in SoX time syntax, and ``"240.0s"`` is an outright parse error.
    """
    output = filepath.with_name(f"{filepath.stem}_spectrogram.png")

    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(filepath), "-vn",
         "-ac", "1", "-sample_fmt", "s16", "-f", "wav", "pipe:1"],
        capture_output=True, check=False)

    if decode.returncode == 0 and decode.stdout:
        spectro = ["spectrogram"]
        if duration_sec > 0:
            spectro += ["-d", f"{duration_sec:.3f}"]   # see docstring: scales the time axis to -x
        sox_result = subprocess.run(
            ["sox", "-t", "wav", "-", "-n",
             *spectro,
             "-x", "1280",   # width in pixels
             "-y", "513",    # height in pixels (2^n + 1 -> fast DFT path in SoX)
             "-z", "120",    # dynamic range in dB
             "-Z", "-20",    # clip ceiling at −20 dB (removes whitewash)
             "-t", filepath.stem,
             "-o", str(output)],
            input=decode.stdout, capture_output=True, check=False)
        if sox_result.returncode == 0 and output.exists():
            return output

    _run([
        "ffmpeg", "-y", "-i", str(filepath), "-vn",
        "-lavfi", "showspectrumpic=s=1280x512:mode=combined:color=fiery:legend=1",
        str(output)
    ])
    # Both renderers can fail (corrupt input, unreadable codec) — report that honestly
    # instead of printing a ✓ with a path to a file that was never written.
    return output if output.exists() else None

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

# Always-on educational context notes (not findings about *this* file). Suppressed
# once the verdict is decided (SUSPICIOUS+); only file-specific caveats survive then.
_GENERIC_CONTEXT_NOTES = (
    "Analog Origins: Vinyl and tape transfers naturally exhibit HF rolloff and higher noise floors; these are not suspicious traits.",
    "Modern Mastering: Audio engineers frequently apply gentle low-pass filters at 19-20 kHz to prevent aliasing distortion.",
    "Transcode Artifacts: Lossless encoders (FLAC/ALAC) will perfectly preserve lossy characteristics if the source material was already degraded prior to encoding.",
)

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
    # Measured codec lowpass walls — (codec, profile, wall Hz, tolerance Hz).
    # Pink-noise fixtures via testdata/make_fixtures.py at 44.1 AND 48 kHz (walls shift
    # with sample rate). Published spec tables are wrong; never replace these with specs.
    # ffmpeg's native AAC ≠ iTunes/FDK cutoffs — tolerance windows absorb encoder spread.
    CODEC_WALLS = (
        ("MP3 (LAME)", "320 kbps", 20220, 150), ("MP3 (LAME)", "320 kbps @48k", 20510, 150),
        ("MP3 (LAME)", "256 kbps", 19530, 150), ("MP3 (LAME)", "256 kbps @48k", 19760, 150),
        ("MP3 (LAME)", "192 kbps", 18840, 150), ("MP3 (LAME)", "192 kbps @48k", 19010, 150),
        ("MP3 (LAME)", "160 kbps", 17460, 150), ("MP3 (LAME)", "128 kbps", 16770, 150),
        ("MP3 (LAME)", "96 kbps", 15410, 150),  ("MP3 (LAME)", "64 kbps", 11270, 250),
        ("AAC", "~192 kbps", 19350, 150),       ("AAC", "~192 kbps @48k", 19560, 150),
        ("AAC", "~128 kbps", 17280, 150),       ("AAC", "~96 kbps", 15860, 180),
        ("Vorbis", "q4 (~128 kbps)", 19000, 150), ("Vorbis", "q4 @48k", 19180, 150),
        ("Vorbis", "q2 (~96 kbps)", 16575, 150),
        ("Opus", "CELT 20 kHz band limit (any bitrate)", 20460, 260),
    )
    # Standard rates a counterfeit "hi-res" file may secretly come from. A resampler
    # leaves its fingerprint at the SOURCE rate's Nyquist — a frequency where natural
    # audio never has features (checked lowest-first so the true origin wins).
    RESAMPLE_SOURCE_RATES = (44100, 48000, 88200, 96000)

    # Codecs that are lossy regardless of container — the .m4a/.mka/.ogg extension says
    # nothing (M4A carries lossy AAC *or* lossless ALAC). mediainfo's Format field does.
    LOSSY_CODECS = {"AAC", "MP3", "MPEG AUDIO", "OPUS", "VORBIS", "WMA", "AC-3",
                    "E-AC-3", "MUSEPACK", "MPC", "ATRAC", "ATRAC3"}

    # Frequencies ≥ this are outside any measured codec lowpass (highest wall: Opus CELT
    # 20,460 Hz; highest foreign Nyquist of a Redbook fake: 22,050 Hz). Walls ABOVE it are
    # DSD→PCM decimation filters (~24–50 kHz) or ultrasonic mastering filters on genuine
    # hi-res — penalising those was the report's false-positive cluster on 96k/192k masters.
    CODEC_CEILING_HZ = 22500.0

    # Time-domain analyses (Hilbert envelopes, cascaded band filters) are capped to
    # this many seconds to bound CPU/RAM on very long files; spectral stats use the full decode.
    TIME_DOMAIN_CAP_S = 180.0

    def __init__(self, filepath: Path, sample_rate: int, channels: int = 2,
                 claimed_duration: float = 0.0, claimed_bitrate_kbps: int = 0,
                 codec: str = ""):
        self.filepath = filepath; self.sample_rate = sample_rate; self.nyquist = sample_rate / 2.0
        self.channels = channels
        self.claimed_duration = claimed_duration
        self.claimed_bitrate_kbps = claimed_bitrate_kbps
        self.codec = codec.upper().strip()
        # Native 1-bit DSD: ffmpeg decodes it through a decimation filter whose ~24-50 kHz
        # wall is conversion physics, not a lossy codec — wall forensics don't apply.
        self.native_dsd = self.codec.startswith("DSD") or filepath.suffix.lower() in {".dff", ".dsf"}
        self.audio_mid: "np.ndarray | None" = None
        self.audio_side: "np.ndarray | None" = None
        # Forward-rfft cache for _fft_band_extract: several detectors band-slice the
        # same capped signal; the forward transform is the expensive half.
        self._rfft_cache: "dict[int, tuple] " = {}

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

    def _compute_frames(self, audio: "np.ndarray", hop: Optional[int] = None) -> "np.ndarray":
        return self._compute_stft(audio, hop=hop)[0]

    def _compute_stft(self, audio: "np.ndarray", hop: Optional[int] = None) -> "tuple[np.ndarray, np.ndarray, int]":
        """Vectorized chunked STFT. Returns (magnitude [frames, bins] float32,
        phase of bins >= 10 kHz [frames, hi_bins] float32, index of first hi bin).
        Magnitude feeds every spectral detector; high-band phase feeds auCDtect entropy.
        hop overrides HOP for detectors that only need subsampled statistics."""
        hop = hop or self.HOP
        n_frames = max(0, (len(audio) - self.WINDOW + hop - 1) // hop)  # == len(range(0, len-WINDOW, hop))
        win = np.hanning(self.WINDOW).astype(np.float32)
        bins = self._freq_bins()
        bin_hz = bins[1] - bins[0]
        hi_start = min(len(bins) - 1, int(10000 / bin_hz))
        idx = np.arange(self.WINDOW)
        mags, phases = [], []
        CHUNK = 512
        for start in range(0, n_frames, CHUNK):
            cnt = min(CHUNK, n_frames - start)
            offs = (np.arange(cnt) + start) * hop
            block = audio[offs[:, None] + idx[None, :]] * win
            # scipy's pocketfft releases the GIL and runs multithreaded
            spec = _srfft(block, axis=1, workers=-1) if _SCIPY_OK else np.fft.rfft(block, axis=1)
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
        # 4x hop: only band-energy MEANS are compared, which converge with far fewer
        # frames — cuts the second STFT to a quarter of the cost.
        side_frames = self._compute_frames(side, hop=self.HOP * 4)

        bin_hz = bins[1] - bins[0]
        idx_10k = int(10000 / bin_hz)
        if idx_10k >= mid_frames.shape[1]: return 0.0

        mid_sub = mid_frames[::4]  # stride matches the side STFT's 4x hop
        n = min(mid_sub.shape[0], side_frames.shape[0])
        mid_hf = mid_sub[:n, idx_10k:]
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
        if frames.shape[0] > 2500:  # bound stats converge long before this — subsample
            frames = frames[:: frames.shape[0] // 2500 + 1]
        ref = frames.max(axis=1, keepdims=True) + 1e-12
        db = 20.0 * np.log10(frames / ref + 1e-12)
        db = np.maximum(db, -110.0)                                  # clamp: decoder numerical residue -> constant
        # float32 running moments: values span [-25, 0] on this scale, so the m2-m1²
        # cancellation error (~1e-4) sits orders below the 0.6 scatter threshold.
        log_power = ((db / 10.0) * np.log(10.0)).astype(np.float32)  # natural-log power scale

        # 5-bin sliding std via running moments (O(n), GIL-free) + 5-bin smoothing —
        # replaces sliding_window_view().std() + median_filter at ~10x the speed.
        m1 = _uniform1d(log_power, 5, axis=1, mode="nearest")
        m2 = _uniform1d(log_power * log_power, 5, axis=1, mode="nearest")
        scatter = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
        scatter = _uniform1d(scatter, 5, axis=1, mode="nearest")

        max_sc = scatter.max(axis=1, keepdims=True)
        thresh = np.minimum(0.6, max_sc * 0.25)
        organic = scatter >= thresh
        has_any = organic.any(axis=1)
        last_idx = organic.shape[1] - 1 - np.argmax(organic[:, ::-1], axis=1)
        bound_bins = np.where(has_any, last_idx, 0)
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
    def _segment_voting(self, audio: "np.ndarray", n_segments: int = 9, wall_hz: float = 16500.0) -> "tuple[int, int, bool, list[tuple[float, float, float, float, float]]]":
        """Cutoff-walls N spread-out 2 s clips and takes a majority vote.
        Returns (walled, valid_total, is_fake, segments) where segments holds
        (offset_seconds, cutoff_hz, cliff_db, void_rel_db, peak_db) per non-silent
        clip — the per-clip data feeds spliced/partial-transcode detection in analyse().
        void_rel_db is the 18.5 kHz→Nyquist band RMS relative to the clip's spectral
        peak: a lossy span decodes to digital zero up there (≈−120 dB) while genuine
        content can never drop below its own dither floor (16-bit ≈ −102 dB) — this
        catches spliced transcodes even when the passage is too dark to show a cliff.
        Silent clips are skipped (their cutoff reads as 0 and would poison the vote).
        wall_hz is adaptive: when the global cutoff has a verified digital void above it
        (or sits on a measured codec wall), the threshold tracks that cutoff instead of
        the fixed AFD 16.5 kHz."""
        if self.nyquist <= wall_hz: return -1, 0, False, []
        total = len(audio)
        seg_samples = int(2.0 * self.sample_rate)
        if total < seg_samples * n_segments: return -1, 0, False, []

        import random
        rng = random.Random(42)  # deterministic
        offsets = []
        step = (total - seg_samples) // (n_segments - 1)
        for i in range(n_segments - 2):
            offsets.append(i * step)
        offsets.append(rng.randint(0, total - seg_samples))
        offsets.append(total - seg_samples)
        offsets.sort()

        walled, segments = 0, []
        win = np.hanning(seg_samples)
        freqs = np.fft.rfftfreq(seg_samples, 1.0 / self.sample_rate)
        bin_hz = freqs[1]
        hi_sel = (freqs >= 18500.0) & (freqs <= self.nyquist - 500.0)
        silent_peak = 10 ** (-50.0 / 20.0)
        for off in offsets:
            clip = audio[off : off + seg_samples].astype(np.float64)
            clip_peak = float(np.max(np.abs(clip)))
            if clip_peak < silent_peak:
                continue
            peak_db = 20 * math.log10(clip_peak)
            mag = np.abs(np.fft.rfft(clip * win))
            ref = mag.max() + 1e-12
            db = 20 * np.log10(mag / ref + 1e-12)
            above = np.where(db > self.CUTOFF_DB)[0]
            cutoff = float(freqs[above[-1]]) if len(above) else 0.0
            # Per-clip cliff: mean dB drop across ±(350-450) Hz around this clip's own
            # cutoff. A transcoded span has a wall here; a naturally dark passage fades.
            cliff = 0.0
            lo_a, lo_b = int((cutoff - 450) / bin_hz), int((cutoff - 350) / bin_hz)
            hi_a, hi_b = int((cutoff + 350) / bin_hz), int((cutoff + 450) / bin_hz)
            if lo_a > 0 and hi_b < len(db) and lo_b > lo_a and hi_b > hi_a:
                cliff = float(db[lo_a:lo_b].mean() - db[hi_a:hi_b].mean())
            void_rel = 0.0
            if hi_sel.any():
                void_rel = float(20 * np.log10(np.sqrt(np.mean((mag[hi_sel] / ref) ** 2)) + 1e-15))
            if cutoff <= wall_hz:
                walled += 1
            segments.append((off / self.sample_rate, cutoff, cliff, void_rel, peak_db))

        valid = len(segments)
        if valid == 0: return -1, 0, False, []
        if valid % 2 == 0: is_fake = walled >= (valid / 2) and walled > 0
        else: is_fake = walled > (valid / 2)
        return walled, valid, bool(is_fake), segments

    def _smooth_envelope(self, x: "np.ndarray", smooth_seconds: float) -> "np.ndarray":
        """Rectified-and-smoothed amplitude envelope, scaled by π/2 so its level tracks
        the Hilbert analytic envelope it replaced (mean |sin| = 2/π). The analytic
        transform cost a full complex-FFT round trip over the signal; |x| + running
        mean is ~10x faster and localizes transients identically at these widths."""
        k = int(smooth_seconds * self.sample_rate) | 1
        return _uniform1d(np.abs(x), k, mode="nearest") * (np.pi / 2)

    def _fft_band_extract(self, x: "np.ndarray", lo: float, hi: float) -> "np.ndarray":
        """Zero-phase brickwall band extraction via FFT masking. IIR skirts (~24 dB/oct)
        leak loud music into a quiet band only ~0.1 octave away; spectral masking gives
        total rejection, which noise-floor forensics above the cutoff depend on.

        Runs in float32 (FFT roundoff is O(eps·log N) ≈ −120 dB — far below the −85 dB
        void threshold) and caches the forward transform per signal length: the void,
        cassette and vinyl rules all band-slice the same capped signal."""
        n = len(x)
        if not _SCIPY_OK:
            X = np.fft.rfft(x.astype(np.float64))
            f = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
            X[(f < lo) | (f > hi)] = 0.0
            return np.fft.irfft(X, n=n)
        key = (n, float(x[0]), float(x[n // 2]), float(x[-1]))
        cached = self._rfft_cache.get(key)
        if cached is None:
            nf = _next_fast_len(n)
            X = _srfft(x.astype(np.float32, copy=False), n=nf, workers=-1)
            f = _srfftfreq(nf, 1.0 / self.sample_rate)
            cached = self._rfft_cache[key] = (X, f, nf)
        X, f, nf = cached
        Y = np.where((f >= lo) & (f <= hi), X, np.complex64(0))
        return _sirfft(Y, n=nf, workers=-1)[:n]

    # -----------------------------------------------------------------------
    # 3-Phase silence / dither / vinyl-surface-noise analyser
    # -----------------------------------------------------------------------
    def _silence_and_vinyl(self, audio: "np.ndarray", cutoff_hz: float, noise_band: "np.ndarray | None" = None,
                           cliff_db: float = 999.0) -> tuple[int, list[str], float, bool, float]:
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
            # The ratio is only meaningful when the music actually CARRIES ultrasonic
            # energy. On dark/acoustic masters (e.g. a 1970 recording) e_music ≈ 0, so
            # tape hiss in the quiet passages balloons the ratio and convicts a genuine
            # file. Require an absolute HF level in the music reference first.
            # hf_energy ≈ 0.1875 × (band RMS)² (hanning Σw² = 0.375N, one-sided ÷2),
            # so 1.2e-8 ≈ a −72 dBFS RMS floor across the 16 kHz+ band.
            if e_music > 1.2e-8:
                silence_ratio = e_silence / (e_music + 1e-12)
                if silence_ratio > 0.3:
                    score += 50
                    reasons.append(f"Codec Noise in Silence: silent passages carry {silence_ratio:.2f}× the music's ultrasonic energy — artificial dither/codec hash, not clean studio silence.")
                    return score, reasons, silence_ratio, False, 0.0
                # NOTE: clean silence is asymmetric evidence. Lossy encoders code digital
                # silence as zeroed frames, so transcodes ALSO have pristine silence.
                # Only dirty silence convicts; the (small, conditional) clean credit is
                # decided in analyse() where wall evidence is known — and we must fall
                # through so the void/vinyl checks still run.

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
                # Two gates before convicting on a void. (1) Codec territory only — a
                # clean void above a 30-48 kHz ultrasonic mastering filter is NORMAL on
                # genuine hi-res masters. (2) The cutoff must be a WALL (>25 dB cliff):
                # a gentle 20-22 kHz mastering rolloff into digital silence is studio
                # practice, not an upscale; codec/SRC walls always fall hard. Both were
                # +20 false-positive modes on the report's genuine 24/96 web releases.
                if cutoff_hz < self.CODEC_CEILING_HZ and cliff_db > 25.0:
                    score += 20
                    reasons.append(f"Digital Upscale Suspect: no noise floor above the cutoff ({energy_db:.1f} dB) under a {cliff_db:.0f} dB wall — analog sources always leave hiss there.")
            else:
                # Lag expresses a physical time offset (~1.13 ms at 44.1k) — keep that
                # time constant at hi-res rates or vinyl hiss at 96/192 kHz reads as
                # correlated and the analog veto never fires.
                autocorr = calculate_autocorrelation(noise_band, lag=max(25, round(50 * sr / 44100)))
                variance = calculate_temporal_variance(noise_band, sr)
                if autocorr < 0.3 and variance < 5.0:
                    vinyl_detected = True
                    score -= 40
                    reasons.append(f"Vinyl Surface Noise: random ({autocorr:.2f} autocorr), temporally stable hiss above the cutoff ({energy_db:.1f} dB) — analog playback signature.")

                    # --- Phase 3: clicks & pops
                    hp = highpass_filter(cap, 1000, sr)
                    env_smooth = self._smooth_envelope(hp, 0.0005)
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

        # float32 throughout: sosfilt/hilbert run ~2x faster and the thresholds here
        # (energy ratios, correlations) are far above single-precision noise.
        cap = np.ascontiguousarray(audio[: int(self.TIME_DOMAIN_CAP_S * sr)], dtype=np.float32)

        # 9A: Pre-echo — MDCT block smearing leaks HF energy *before* sharp transients
        env_smooth = self._smooth_envelope(cap, 0.001)
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
            # Direct dot-product Pearson on non-overlapping 5 s segments — corrcoef()
            # stacked/copied each pair and the 50% overlap added nothing to the median.
            for i in range(0, len(band_a) - seg_len + 1, max(1, seg_len)):
                sa, sb = band_a[i : i + seg_len], band_b_inv[i : i + seg_len]
                ma, mb = float(sa.mean()), float(sb.mean())
                va, vb = float(sa.var()), float(sb.var())
                if va > 1e-12 and vb > 1e-12:
                    cov = float(np.dot(sa, sb)) / seg_len - ma * mb
                    corrs.append(abs(cov / math.sqrt(va * vb)))
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
    def _cassette_source(self, audio: "np.ndarray", frames: "np.ndarray", bins: "np.ndarray",
                         cutoff_hz: float, cutoff_std: float, mp3_detected: bool) -> tuple[int, list[str], bool]:
        """Returns (score, reasons, hiss_found). The veto in analyse() additionally
        requires hiss_found — a cassette without tape hiss does not exist, and the
        slope/flutter rules alone must not disarm the segment vote on real transcodes."""
        score, reasons, hiss_found = 0, [], False
        if cutoff_hz >= 19000: return 0, reasons, False
        sr = self.sample_rate
        cap = np.ascontiguousarray(audio[: int(60.0 * sr)], dtype=np.float32)

        # 11A: constant tape hiss above the musical cutoff (FFT brickwall — no music leakage)
        upper_limit = min(20000.0, sr / 2 - 100)
        noise_lo = cutoff_hz + 1000 if cutoff_hz < 16000 else cutoff_hz + 500
        if upper_limit > noise_lo:
            noise_sig = self._fft_band_extract(cap, noise_lo, upper_limit)
            noise_db = 20 * math.log10(float(np.std(noise_sig)) + 1e-12)
            autocorr = calculate_autocorrelation(noise_sig, lag=max(50, round(100 * sr / 44100)))
            if noise_db > -55.0 and autocorr < 0.2:
                score += 30
                hiss_found = True
                reasons.append(f"R11A: Tape hiss present above cutoff ({noise_db:.1f} dB, random autocorr {autocorr:.2f}).")

        # 11B: natural magnetic-tape roll-off slope across 12-18 kHz, read straight from
        # the cached STFT (slope is ref-invariant; replaces 20 sequential bandpass runs)
        bin_hz = bins[1] - bins[0]
        avg = frames.mean(axis=0)
        db_spec = 20.0 * np.log10(avg / (avg.max() + 1e-12) + 1e-12)
        res = []
        for f in np.linspace(12000, 18000, 20):
            if f + 250 < sr / 2:
                lo_i, hi_i = int((f - 250) / bin_hz), int((f + 250) / bin_hz)
                res.append(float(db_spec[lo_i:hi_i].mean()) if hi_i > lo_i else -120.0)
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

        return max(0, score), reasons, hiss_found

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

    def _resample_check(self, frames: "np.ndarray", bins: "np.ndarray") -> "tuple[int, str, float] | None":
        """Sample-rate provenance: hunt for resampler fingerprints at foreign Nyquists.

        Upsampling (e.g. 44.1 kHz Red Book -> "24/48") leaves one of two signatures
        at the source rate's Nyquist:
          - "wall":  the spectrum ends in a cliff exactly there (clean resampler);
          - "notch": a deep hole exactly there with imaging/injected energy ABOVE it.
                     The energy above pushes the measured HF cutoff to full bandwidth
                     and defeats every cutoff-based detector — but nothing natural
                     carves a 20+ dB hole at precisely 22,050 Hz;
          - "mirror": a weak anti-imaging filter (ffmpeg swr default) passes the
                     aliased images almost unattenuated, so the spectrum continues
                     smoothly across the fold — but the content above fn is a
                     MIRROR image of the content below it, exposed by per-frame
                     magnitude correlation of the two bands.
        Returns (source_rate, mode, depth_db_or_corr) or None.
        """
        if frames.shape[0] == 0: return None
        avg = frames.mean(axis=0).astype(np.float64)
        db = 20.0 * np.log10(avg / (avg.max() + 1e-12) + 1e-12)
        bin_hz = bins[1] - bins[0]

        def band(lo_hz: float, hi_hz: float) -> float:
            i0, i1 = max(0, int(lo_hz / bin_hz)), min(len(db), int(hi_hz / bin_hz) + 1)
            return float(db[i0:i1].mean()) if i1 > i0 else -200.0

        for rate in self.RESAMPLE_SOURCE_RATES:
            fn = rate / 2.0
            if fn + 1200.0 > self.nyquist - 200.0: continue   # need >=750 Hz of room above
            edge = band(fn - 900, fn - 350)                   # music just below the transition band
            if edge < -80.0: continue                          # nothing alive near this Nyquist — no evidence either way
            notch = band(fn - 150, fn + 250)                  # resampler transition/void straddles fn
            above = band(fn + 450, min(fn + 2000, self.nyquist - 200))   # imaging/noise flank
            ceiling = band(fn + 450, self.nyquist - 200)      # everything above (wall mode)
            if edge - notch >= 20.0 and above - notch >= 20.0:
                return rate, "notch", min(edge, above) - notch
            if ceiling < edge - 40.0 and ceiling < -90.0:
                return rate, "wall", edge - ceiling
            # Mirror mode: correlate the band below fn against the band above it,
            # per frame (bin fn-x pairs with bin fn+x), with a guard band around
            # the fold to skip the resampler's transition region. Aliased images
            # carry conjugated coefficients, so frame magnitudes mirror EXACTLY
            # per bin — sample the mirror positions with fractional-bin
            # interpolation (the fold sits between bin centres) and do NOT
            # smooth: smoothing mixes in uncorrelated secondary images
            # (fractional resamplers fold k>=2 images into the same band) and
            # dilutes the correlation. Measured: ffmpeg swr 44.1->48 fixture
            # 0.49, genuine full-band noise 0.003.
            c, g = int(round(fn / bin_hz)), int(300.0 / bin_hz)
            half = min(int(1800.0 / bin_hz), len(db) - 4 - c - g, c - g)
            if half * bin_hz >= 700.0 and above > -85.0:
                lo_idx = np.arange(c - g - half, c - g)
                target = (fn * 2.0 / bin_hz) - lo_idx           # fractional mirror positions
                t0 = np.floor(target).astype(np.int64); tfrac = target - t0
                hi_lin = frames[:, t0] * (1.0 - tfrac) + frames[:, t0 + 1] * tfrac
                lo_band = np.log(frames[:, lo_idx].astype(np.float64) + 1e-12)
                hi_band = np.log(hi_lin.astype(np.float64) + 1e-12)
                lo_band -= lo_band.mean(axis=1, keepdims=True)
                hi_band -= hi_band.mean(axis=1, keepdims=True)
                denom = np.sqrt((lo_band * lo_band).sum(axis=1) * (hi_band * hi_band).sum(axis=1)) + 1e-12
                mirror_corr = float(np.mean((lo_band * hi_band).sum(axis=1) / denom))
                if mirror_corr > 0.35:
                    return rate, "mirror", mirror_corr
        return None

    def _codec_fingerprint(self, cutoff_hz: float) -> "tuple[str, str, int] | None":
        """Nearest measured codec wall within tolerance → (codec, profile, wall_hz).
        A cutoff that lands exactly on a measured encoder lowpass is a signature;
        an arbitrary mastering filter almost never does."""
        if cutoff_hz <= 0 or cutoff_hz >= self.nyquist * 0.98: return None
        best = None
        for codec, profile, hz, tol in self.CODEC_WALLS:
            d = abs(cutoff_hz - hz)
            if d <= tol and (best is None or d < best[0]):
                best = (d, codec, profile, hz)
        return (best[1], best[2], best[3]) if best else None

    @staticmethod
    def _is_fake_hires_bandwidth(sample_rate: int, nyquist: float, cutoff_hz: float,
                                 cliff_depth: float, void_db: float, void_measured: bool) -> bool:
        """Fake hi-res by insufficient bandwidth: a high-rate container whose content
        ends in a hard wall far below its own Nyquist, with a digital void above.

        Catches Redbook -> lossy -> hi-res chains where the codec lowpass erased the
        foreign-Nyquist tell (notch/mirror/wall), so the cutoff-based and resample
        detectors all go blind. Gated hard so genuine high-rate masters (content to
        Nyquist) and gentle/analog rolloffs (shallow cliff, audible hiss above) never
        trip it: only >=88.2 kHz, cutoff <60% Nyquist, >25 dB cliff, measured void
        below -80 dB qualifies. Thresholds live here so they're tunable in one place."""
        return (sample_rate >= 88200 and 0 < cutoff_hz < nyquist * 0.6
                and cliff_depth > 25.0 and void_measured and void_db < -80.0)

    def _verdict(self, main_score: int, net_score: int, cutoff_hz: float, dsd_detected: bool,
                 cassette: bool = False, vinyl: bool = False, resampled_from: int = 0,
                 fake_hires: str = "") -> tuple[str, str, list[str]]:
        caveats = list(_GENERIC_CONTEXT_NOTES)
        ext = self.filepath.suffix.lower()
        # Native lossy is decided by the CODEC, not the extension — .m4a carries lossy
        # AAC or lossless ALAC, and Opus hides in .m4a/.mka too (the report's bypass).
        if self.codec in self.LOSSY_CODECS or ext in {".mp3", ".aac", ".ogg", ".opus", ".wma"}:
            fmt_label = self.codec if len(self.codec) <= 3 else self.codec.title()
            if not fmt_label: fmt_label = ext.upper().lstrip(".")
            elif f".{self.codec.lower()}" != ext:
                fmt_label += f" in {ext.upper()}"
            fp = self._codec_fingerprint(cutoff_hz)
            # The codec is KNOWN here — a nearest-wall match from a different codec
            # (e.g. an Opus cutoff brushing the LAME 320 wall) is noise, not signal.
            if fp and self.codec and self.codec not in fp[0].upper(): fp = None
            fp_match = f" — matches measured {fp[0]} {fp[1]} encoder profile" if fp else ""
            sentence = f"ℹ Natively Lossy Format ({fmt_label}){fp_match}"
            if not fp_match and net_score >= 6: sentence += " — severe degradation detected."
            return "CAUTION", sentence, []
        if self.native_dsd:
            caveats.append("Native DSD (1-bit) stream — the analysed PCM came through a DSD decimation filter; its ~24–50 kHz wall and shaped ultrasonic noise are conversion physics, not codec damage.")
            return "GENUINE", "ℹ Native DSD source — PCM-domain transcode forensics do not apply to the 1-bit stream", caveats
        if not _SCIPY_OK: caveats.append("scipy not installed — advanced DSP suite skipped; verdict relies on the base spectral engine only.")
        if dsd_detected: caveats.append("DSD transcode detected. Ultrasonic noise inflates entropy and HF scores.")
        if cassette: caveats.append("Cassette source profile matched — HF limitations are analog tape physics, not codec damage.")
        if vinyl: caveats.append("Vinyl surface noise detected — rolloff and noise floor traits are analog, not codec damage.")
        if resampled_from: caveats.append("Sample-rate upscale detected — bit-depth verification cannot see through resampling (interpolation regenerates the low-order bits), so a 'verified' bit-depth reading does not prove source depth.")
        if fake_hires: caveats.append("Fake hi-res by bandwidth — the container's high sample rate carries no matching high-frequency content. Bit-depth verification is also unreliable here; the upsample regenerated the low-order bits.")
        if main_score >= 86: return "LIKELY_LOSSY", "✗  Lossy transcode detected — fake lossless (high certainty)", caveats
        elif resampled_from: return "SUSPICIOUS", f"⚠  Sample-rate counterfeit — upsampled from {resampled_from / 1000:g} kHz (fake hi-res)", caveats
        elif fake_hires: return "SUSPICIOUS", f"⚠  Fake hi-res — {fake_hires} (upsampled)", caveats
        elif main_score >= 55: return "SUSPICIOUS", "⚠  Strong lossy indicators — probable transcode", caveats
        elif main_score >= 31: return "CAUTION", "~  Minor spectral quirks — possibly legitimate", caveats
        elif main_score >= 11: return "LIKELY_GENUINE", "✓  Consistent with genuine lossless source", caveats
        else: return "GENUINE", "✓  Strong evidence of authentic lossless source", caveats

    def analyse(self, max_seconds: Optional[float] = None, status=None) -> SpectralAnalysis:
        st = status if status is not None else (lambda _msg: None)
        result = SpectralAnalysis()
        result.scipy_available = _SCIPY_OK
        if not _NUMPY_OK:
            result.primary_verdict = "numpy not installed"; result.verdict_label = "INCONCLUSIVE"; return result
        if not _SCIPY_OK:
            print("Warning: scipy not installed — advanced forensic suite (auCDtect, vinyl/cassette, "
                  "psychoacoustic tests) disabled. pip install scipy", file=sys.stderr)

        # Single decode: stereo when available (mid feeds every mono detector, side feeds joint-stereo forensics)
        st("decoding")
        audio, side = None, None
        if self.channels >= 2 and (pair := self._decode_stereo(max_seconds)) is not None:
            audio, side = pair
        if audio is None:
            audio = self._decode_audio(max_seconds)
        # Retained so build_report can derive clipping/phase/silence without re-decoding
        self.audio_mid, self.audio_side = audio, side
        if audio is None or len(audio) < self.WINDOW * 2:
            result.primary_verdict = "Could not decode audio"; result.verdict_label = "INCONCLUSIVE"; return result

        st("STFT")
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

        st("spectral metrics")
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

        # Rule: sample-rate provenance — upsample / fake hi-res detection (numpy-only,
        # runs even without scipy). Digital evidence: immune to the analog vetoes.
        st("resample check")
        resample_hit = self._resample_check(frames, bins)
        if resample_hit:
            res_rate, res_mode, res_depth = resample_hit
            fn_src = res_rate // 2
            result.resample_src_rate = res_rate
            if res_mode == "notch":
                result.resample_detected = f"upsampled from {res_rate / 1000:g} kHz — imaging notch at {fn_src:,} Hz ({res_depth:.0f} dB deep)"
                lossy_ev.append(f"Sample-Rate Upscale: {res_depth:.0f} dB spectral notch at exactly {fn_src:,} Hz — the {res_rate / 1000:g} kHz Nyquist. The file was upsampled; energy above the notch is resampler imaging/noise, not music, and the container rate ({self.sample_rate / 1000:g} kHz) is counterfeit.")
            elif res_mode == "mirror":
                result.resample_detected = f"upsampled from {res_rate / 1000:g} kHz — aliased mirror around {fn_src:,} Hz (corr {res_depth:.2f})"
                lossy_ev.append(f"Sample-Rate Upscale: the spectrum above {fn_src:,} Hz is a mirror image of the spectrum below it (per-frame correlation {res_depth:.2f}) — aliasing imaging from a low-quality upsampler. The container rate ({self.sample_rate / 1000:g} kHz) is counterfeit.")
            else:
                result.resample_detected = f"upsampled from {res_rate / 1000:g} kHz — wall at exactly {fn_src:,} Hz"
                lossy_ev.append(f"Sample-Rate Upscale: spectrum ends in a hard wall at exactly {fn_src:,} Hz — the {res_rate / 1000:g} kHz Nyquist — with digital void above. The container rate ({self.sample_rate / 1000:g} kHz) is counterfeit.")
            main += 45

        if _SCIPY_OK:
            # Rule: Fakin' the Funk header integrity
            st("header integrity")
            decoded_dur = len(audio) / self.sample_rate if max_seconds is None else 0.0
            dur_mm, br_mm, hdr_reasons = self._check_header_integrity(decoded_dur)
            result.header_duration_mismatch, result.header_bitrate_mismatch = dur_mm, br_mm
            if dur_mm: main += 20
            if br_mm: main += 25
            lossy_ev.extend(hdr_reasons)

            # Rule: psychoacoustic artefacts (pre-echo / aliasing / MP3 subband comb)
            st("psychoacoustic tests")
            psy_score, psy_ev, preecho, aliasing, mp3_noise = self._psychoacoustic_artifacts(audio, frames, bins, cutoff_hz, mp3_profile_match)
            result.preecho_pct, result.aliasing_corr, result.mp3_noise_pattern_detected = preecho, aliasing, mp3_noise
            main += psy_score
            lossy_ev.extend(psy_ev)

            # Rule 11: cassette source profiler (veto — analog tape, not codec damage)
            st("cassette profile")
            cass_score, cass_ev, cass_hiss = self._cassette_source(audio, frames, bins, cutoff_hz, cutoff_std, mp3_noise)
            result.cassette_score = cass_score
            cassette_detected = cass_score >= 30 and cass_hiss
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

            # Rule: fake hi-res by insufficient bandwidth. A >=88.2 kHz container whose
            # content ends in a hard wall far below its OWN Nyquist, with a digital void
            # above, was upsampled from a lower rate — even when the foreign-Nyquist tell
            # (notch/mirror/wall) was erased by a codec lowpass sitting below it. Redbook
            # -> AAC -> 24/96 is the canonical miss: AAC's ~20 kHz lowpass kills the
            # 22,050 Hz signature, so _resample_check goes blind on a dead band. Genuine
            # hi-res carries content/noise well past 24 kHz; a 20 kHz ceiling in a 96k
            # file is counterfeit. Gated hard (hi-res rate + cutoff <60% Nyquist + steep
            # cliff + digital void) so dark analog masters and gentle LPFs never trip it.
            if not resample_hit and not self.native_dsd and self._is_fake_hires_bandwidth(
                    self.sample_rate, self.nyquist, cutoff_hz, cliff_depth, void_db, noise_band is not None):
                main += 20
                result.fake_hires = f"{self.sample_rate / 1000:g} kHz container but bandwidth ends at {cutoff_hz / 1000:.1f} kHz"
                lossy_ev.append(f"Fake Hi-Res Bandwidth: the {self.sample_rate / 1000:g} kHz container carries nothing above {cutoff_hz / 1000:.1f} kHz — a {cliff_depth:.0f} dB wall into a digital void ({void_db:.0f} dB) at {cutoff_hz / self.nyquist * 100:.0f}% of Nyquist. Genuine high-rate masters reach far higher; the stream was upsampled and the sample rate is counterfeit.")

            # Rule: AFD PRO segment voting (skipped under cassette veto).
            # Adaptive wall: a steep cliff (>30 dB) with a verified digital void above it
            # IS a codec wall wherever it sits — track it instead of the fixed 16.5 kHz so
            # high-cutoff encoders (LAME 320 walls at ~20.2 kHz) cannot slip past the vote.
            # A cutoff sitting exactly on a measured codec wall (fingerprint) arms it too:
            # AAC's own residue above its cutoff can defeat the void check, but a 64 dB
            # cliff at a measured encoder frequency is a signature, not mastering.
            # Natural fades have shallow cliffs and dark analog sources leave hiss, so
            # neither can arm this.
            st("segment voting")
            fingerprint = self._codec_fingerprint(cutoff_hz)
            void_verified = noise_band is not None and void_db < -85.0
            wall_hz = 16500.0
            # A wall sitting exactly on a foreign Nyquist is resample evidence, not a
            # codec wall — arming the vote on it would mislabel a lossless upsample
            # as "whole-file lossy ancestry" (the resample rule already scored it).
            wall_is_resample = resample_hit is not None and resample_hit[1] == "wall"
            # Ceiling gate: only codec-territory walls (<22.5 kHz) may arm the adaptive
            # vote. DSD decimation filters (~24-50 kHz) and steep ultrasonic mastering
            # filters on genuine hi-res sit above it — both are walls with voids, and
            # arming on them convicts genuine files (the DSD64 false LIKELY_LOSSY).
            if (cliff_depth > 30.0 and (void_verified or fingerprint)
                    and not wall_is_resample and cutoff_hz < self.CODEC_CEILING_HZ):
                wall_hz = max(wall_hz, cutoff_hz + 400.0)
            # Probe density adapts to duration (one probe ≈ every 15 s, 9-36 probes):
            # 9 fixed probes on a 5-minute track stride ~38 s and step right over a
            # 30 s spliced transcode (the report's spliced_mp3 miss).
            n_seg = int(min(36, max(9, (len(audio) / self.sample_rate) // 15)))
            seg_walled, seg_total, seg_fail, seg_data = self._segment_voting(audio, n_segments=n_seg, wall_hz=wall_hz)
            result.segment_walled, result.segment_total, result.segment_wall_hz = seg_walled, seg_total, wall_hz
            if seg_fail and not cassette_detected:
                main += 55
                lossy_ev.append(f"Segment Vote FAILED: {seg_walled}/{seg_total} sampled 2s clips are frequency-walled at ≤{wall_hz / 1000:.1f} kHz — consistent whole-file lossy ancestry.")
            elif not cassette_detected and seg_data:
                # Spliced/partial transcode: clips walled far below the global spectrum
                # WITH a per-clip cliff are mixed lossy ancestry even when the majority
                # of the file is clean. The cliff requirement keeps naturally dark or
                # quiet passages (gradual fades, no wall) out.
                anomaly_ceiling = min(cutoff_hz - 2000.0, self.nyquist * 0.85)
                # Two independent views of a spliced lossy span:
                #  - cliff: the clip's spectrum ends in a >25 dB wall (loud transcoded music);
                #  - void:  the clip's 18.5k+ band is ≥110 dB below its own peak while music
                #    plays. Nothing genuine does that — 16-bit dither alone sits ~−102 dB rel
                #    (measured: spliced MP3 clips −120, genuine quiet fades −95..−102). This
                #    catches splices whose passages are too dark for the wall to show.
                #    Gated on a full-band global spectrum so an honest lowpassed master
                #    (every clip "voided") can't trip it, and on clip peak > −40 dBFS so
                #    fade-outs into dithered noise floor don't count.
                full_band_global = cutoff_hz > self.nyquist * 0.93
                anom = {}
                for t, c, cl, vr, pk in seg_data:
                    is_cliff = 0 < c < anomaly_ceiling and cl > 25.0
                    is_void = (full_band_global and 11000.0 < c < anomaly_ceiling
                               and vr < -110.0 and pk > -40.0)
                    if is_cliff or is_void:
                        anom[t] = (t, c, cl, vr, "void" if is_void else "cliff")
                anomalous = sorted(anom.values())
                void_backed = any(kind == "void" for *_, kind in anomalous)
                if len(anomalous) == 1:
                    # A single walled probe is conviction-grade only when its wall is
                    # unmistakable: a >35 dB cliff sitting ON a measured codec lowpass.
                    # A naturally dark 2 s passage has neither.
                    t1, c1, cl1, vr1, _ = anomalous[0]
                    seg_fp1 = self._codec_fingerprint(c1)
                    if cl1 > 35.0 and seg_fp1:
                        main += 25
                        result.segment_map = [f"{int(t1 // 60):02d}:{int(t1 % 60):02d} → walled at {c1 / 1000:.1f} kHz (cliff {cl1:.0f} dB)"]
                        lossy_ev.append(f"Partial Transcode (single region): one sampled clip at {int(t1 // 60):02d}:{int(t1 % 60):02d} is frequency-walled at {c1 / 1000:.1f} kHz with a {cl1:.0f} dB cliff sitting on the measured {seg_fp1[0]} {seg_fp1[1]} lowpass — a spliced lossy span inside otherwise full-band content.")
                if len(anomalous) >= 2:
                    # Confidence scales with coverage; a digital void under the walled
                    # clips (impossible naturally) or a codec fingerprint on their median
                    # cutoff upgrades it further.
                    seg_fp = self._codec_fingerprint(float(np.median([c for _, c, *_ in anomalous])))
                    bonus = 40 if len(anomalous) >= 4 else 30
                    if void_backed: bonus += 25
                    if seg_fp: bonus += 15
                    main += bonus
                    result.segment_map = [
                        f"{int(t // 60):02d}:{int(t % 60):02d} → walled at {c / 1000:.1f} kHz ({'digital void above' if kind == 'void' else f'cliff {cl:.0f} dB'})"
                        for t, c, cl, vr, kind in anomalous]
                    regions = ", ".join(f"{int(t // 60):02d}:{int(t % 60):02d}" for t, *_ in anomalous[:6])
                    fp_note = f" — walls sit on a measured codec lowpass (nearest profile: {seg_fp[0]} {seg_fp[1]})" if seg_fp else ""
                    void_note = " with a digital void above the wall (impossible in genuine content)" if void_backed else ""
                    lossy_ev.append(f"Partial Transcode: {len(anomalous)}/{len(seg_data)} sampled clips are frequency-walled{void_note} while the rest of the file is full-band — spliced or partially transcoded content (walled at {regions}){fp_note}.")

            # Rule: silence dither / vinyl noise / clicks (3-phase)
            if not cassette_detected:
                st("silence & vinyl analysis")
                sil_score, sil_ev, sil_ratio, vinyl_detected, clicks = self._silence_and_vinyl(audio, cutoff_hz, noise_band=noise_band, cliff_db=cliff_depth)
                result.silence_ratio, result.vinyl_noise_detected, result.vinyl_clicks_per_min = sil_ratio, vinyl_detected, clicks
                main += sil_score
                (lossy_ev if sil_score > 0 else natural_ev).extend(sil_ev)
                # Clean silence is weak, asymmetric evidence (lossy encoders zero out
                # silence too) — credit it only when the spectrum is full-bandwidth and
                # no wall evidence exists, and keep it small.
                # (a resampled file's clean silence proves nothing about provenance)
                if 0 <= sil_ratio < 0.15 and not seg_fail and wall_hz <= 16500.0 and cutoff_hz > self.nyquist * 0.85 and not resample_hit:
                    main -= 30
                    natural_ev.append(f"Clean Silence Floor: silent passages are spectrally clean (ratio {sil_ratio:.2f}) in a full-bandwidth spectrum — consistent with an unmolested lossless master.")

            # Rule: codec wall fingerprint — the cutoff lands on a measured encoder
            # lowpass (CODEC_WALLS) with a verified void or deep cliff behind it.
            # Arbitrary mastering filters almost never hit these exact frequencies;
            # this is what separates a LAME 320 wall from a legit steep mastering LPF.
            if fingerprint and (void_verified or cliff_depth > 30.0) and not cassette_detected and not vinyl_detected:
                fp_codec, fp_profile, fp_hz = fingerprint
                result.codec_fingerprint = f"{fp_codec} {fp_profile}"
                main += 10
                lossy_ev.append(f"Codec Wall Fingerprint: cutoff {cutoff_hz:,.0f} Hz sits on the measured {fp_codec} {fp_profile} lowpass ({fp_hz:,} Hz) — an encoder signature, not mastering.")

            # Rule: auCDtect statistical bound frequency & high-band phase entropy
            st("auCDtect statistics")
            auc_avg, auc_prob, auc_phase = self._aucdtect_features(frames, phase_act, bins)
            result.auc_avg_bound_freq, result.auc_prob_bound_freq, result.auc_phase_entropy = auc_avg, auc_prob, auc_phase
            if self.sample_rate >= 40000 and 0 < auc_avg < 16500 and not cassette_detected and not vinyl_detected:
                main += 25
                lossy_ev.append(f"auCDtect Bound Collapse: spectral scatter dies at {auc_avg:,.0f} Hz on average — the statistical void of a lossy codec.")
            # Quantized HF phase is codec evidence only when the cutoff is in codec
            # territory AND ends in a wall — genuine masters with gentle 20-48 kHz
            # mastering filters have noisy HF phase too (the report's +10 false
            # positive on 24/96 files). Corroborating evidence, never standalone.
            if (auc_phase > 4.5 and cutoff_hz < self.nyquist * 0.85
                    and cutoff_hz < self.CODEC_CEILING_HZ and cliff_depth > 25.0):
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
        label, sentence, caveats = self._verdict(main, net_score, cutoff_hz, dsd_detected, cassette_detected, vinyl_detected,
                                                 resampled_from=result.resample_src_rate, fake_hires=result.fake_hires)

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
    t_start = time.perf_counter()
    name = filepath.name
    _Status.update(name, "probing metadata")
    tags, tech = extract_mediainfo(filepath)

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

    # Native DSD reports its 1-bit rate (2.8/5.6 MHz) — decoding at that rate is
    # pathological (gigabytes of float32). Decimate to 88.2 kHz: the audible band plus
    # enough ultrasonic headroom for the DSD noise-shaping scan to see the 30 kHz+ rise.
    native_dsd = tech.codec.upper().startswith("DSD") or filepath.suffix.lower() in {".dff", ".dsf"}
    if native_dsd and sample_rate > 96000:
        sample_rate = 88200

    engine = SpectralEngine(filepath, sample_rate, channels=channels,
                            claimed_duration=tech.duration_sec, claimed_bitrate_kbps=claimed_bitrate,
                            codec=tech.codec)

    # Subprocess-bound extractors run concurrently while the DSP engine crunches
    # on the main thread (numpy/scipy release the GIL for the heavy operations).
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_loud = pool.submit(extract_loudness, filepath)
        f_sox = pool.submit(extract_sox_stats, filepath)
        f_spec = pool.submit(generate_spectrogram, filepath, tech.duration_sec)
        f_bits = pool.submit(check_bit_depth_authenticity, filepath, claimed_depth, tech.duration_sec, sample_rate, channels)
        spectral = engine.analyse(max_seconds=fast_secs, status=lambda s: _Status.update(name, s))
        _Status.update(name, "waiting on loudness/spectrogram")
        lp, dr = f_loud.result()
        sox = f_sox.result()
        spec_path = f_spec.result()
        bit_auth = f_bits.result()
    _Status.update(name, "finalizing")

    auth = AuthenticityReport()
    auth.spectral = spectral
    auth.spectral_cutoff_hz = spectral.cutoff_hz_str
    auth.spectral_cutoff_verdict = spectral.primary_verdict
    auth.lpf_detected = spectral.lpf_detected
    auth.lpf_cutoff_hz = spectral.lpf_cutoff_str
    auth.cassette_rip_detected = spectral.cassette_score >= 30
    auth.vinyl_rip_detected = spectral.vinyl_noise_detected
    auth.side_channel_analysis = f"{spectral.side_anomaly_score:.3f} {spectral.side_interp}" if channels >= 2 else "mono — no side channel"
    if spectral.header_duration_mismatch or spectral.header_bitrate_mismatch:
        kinds = [k for k, f in (("duration", spectral.header_duration_mismatch), ("bitrate", spectral.header_bitrate_mismatch)) if f]
        auth.header_integrity = f"⚠ Header {' & '.join(kinds)} mismatch — forged or truncated stream"
    elif spectral.scipy_available and spectral.verdict_label != "INCONCLUSIVE":
        auth.header_integrity = "✓ Container header matches decoded stream"
    auth.bit_depth_authentic = bit_auth
    if native_dsd:
        auth.bit_depth_authentic = "DSD 1-bit stream — PCM trailing-zero analysis not applicable"
    auth.encoder_trace = detect_encoder_trace(tags, tech, filepath)
    if mqa_note := detect_mqa(tags, tech, filepath):
        spectral.caveats.append(mqa_note)
        auth.mqa_detected = True
    # Byproducts of the engine's decode — no extra ffmpeg processes
    auth.phase_correlation, auth.phase_verdict = measure_phase_correlation(engine.audio_mid, engine.audio_side, sample_rate)
    auth.clipped_samples, auth.clipping_verdict = detect_clipping(engine.audio_mid, engine.audio_side)
    auth.silence_total_pct, auth.silence_sections = map_silence(engine.audio_mid, sample_rate, tech.duration_sec)
    auth.rg_stored, auth.rg_measured_lufs, auth.rg_delta, auth.rg_verdict = audit_replaygain(tags, lp.lufs_integrated)
    if not lp.noise_floor_db:
        # astats reports nan/inf noise floor on some content — measure it ourselves:
        # 5th percentile of per-100ms RMS across the decoded track.
        lp.noise_floor_db = _noise_floor_from_audio(engine.audio_mid, sample_rate)
    engine.audio_mid = engine.audio_side = None  # release decode buffers

    report = ForensicReport(filepath=filepath, tags=tags, technical=tech, sox_stats=sox, loudness=lp, authenticity=auth, dr_score=dr, spectrogram_path=spec_path)
    report.analysis_seconds = time.perf_counter() - t_start
    _Status.done(name)
    return report

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
    known_rows = [_kv("Title", t.title), _kv("Artist", t.artist), _kv("Album", t.album), _kv("Album Artist", t.album_artist), _kv("Year", t.date), _kv("BPM", t.bpm), _kv("Comment", t.comments), _kv("Rip Quality", t.comment_quality)]
    printed_any = False
    for row in known_rows:
        if row: print(row); printed_any = True
    if t.other:
        for key in sorted(t.other):
            val = t.other[key]
            if len(val) > 70: val = val[:70] + "…"
            print(_kv(key[:25], val)); printed_any = True
    if not printed_any:
        print(f"  {_c(C.GREY, 'No tags found')}")
    print(_section("TECHNICAL"))
    for row in [_kv("Encoding", tec.sample_encoding), _kv("Format Profile", tec.format_profile), _kv("Bit Rate", tec.bit_rate), _kv("Sample Rate", _hz_label(tec.sample_rate)), _kv("Channels", _channel_label(tec.channels)), _kv("Precision", tec.precision), _kv("Compression", tec.compression_mode), _kv("Writing Library", tec.writing_library)]:
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
            fp_val = _c(C.RED, f"⚠ {sp.codec_fingerprint} wall profile") if sp.codec_fingerprint else _c(C.GREEN, "✓ cutoff matches no known encoder wall")
            if sp.resample_detected: res_val = _c(C.RED, f"⚠ {sp.resample_detected}")
            elif sp.fake_hires: res_val = _c(C.ORANGE, f"⚠ fake hi-res — {sp.fake_hires} (upsampled; codec lowpass erased the Nyquist tell)")
            else: res_val = _c(C.GREEN, "✓ no foreign-Nyquist resampler artifacts")
            rows_adv = [
                _kv("Header Integrity", _c(C.RED, "⚠ header/stream mismatch — forged or truncated") if hdr_mm else _c(C.GREEN, "✓ container matches decoded stream")),
                _kv("Codec Fingerprint", fp_val),
                _kv("Resample Check", res_val),
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
            if sp.segment_map:
                print(f"    {_c(C.ORANGE, '⚠ Partially transcoded regions:')}")
                for seg_line in sp.segment_map[:6]:
                    print(f"      {_c(C.GREY, '→')} {_c(C.WHITE, seg_line)}")
        else:
            print(f"  {_c(C.YELLOW, '⚠ scipy not installed — advanced forensic suite skipped (pip install scipy)')}")

        # Once the verdict is decided (SUSPICIOUS+), the green "Natural indicators"
        # read as if they argue against it and the always-on context notes are noise —
        # suppress both, keeping only file-specific caveats under a "Caveats" header.
        decided = sp.verdict_label in ("SUSPICIOUS", "LIKELY_LOSSY")
        if sp.evidence:
            print(f"\n  {_c(C.DIM + C.ORANGE, 'Lossy indicators')}")
            for e in sp.evidence: print(f"    {_c(C.GREY, '·')} {_c(C.WHITE, e)}")
        if sp.natural_evidence and not decided:
            print(f"\n  {_c(C.DIM + C.GREEN, 'Natural indicators')}")
            for n in sp.natural_evidence: print(f"    {_c(C.GREY, '·')} {_c(C.GREEN, n)}")
        notes = [cv for cv in sp.caveats if not decided or cv not in _GENERIC_CONTEXT_NOTES]
        if notes:
            print(f"\n  {_c(C.DIM + C.GREY, 'Caveats' if decided else 'Context notes')}")
            for cv in notes: print(f"    {_c(C.GREY, '·')} {_c(C.DIM + C.WHITE, cv)}")
    else:
        for row in [_kv("HF Cutoff", auth.spectral_cutoff_hz), _kv("Spectral Verdict", auth.spectral_cutoff_verdict), _kv("LPF Detected", ("⚠ YES — cutoff at " + auth.lpf_cutoff_hz) if auth.lpf_detected else "✓ No LPF detected")]:
            if row: print(row)

    print(_subsection("Source Integrity"))
    source_flags = []
    if auth.cassette_rip_detected: source_flags.append("cassette tape")
    if auth.vinyl_rip_detected: source_flags.append("vinyl")
    bd_text = auth.bit_depth_authentic
    # Any transcode/upsample chain repopulates the low-order bits (float decode +
    # re-quantization, or resampling interpolation), so used-bits analysis is blind
    # to the source depth and the noise floor (if exposed at all) is the encoder's,
    # not the source's. Don't render a green pass — that contradicts the lossy/upscale
    # finding above. Keyed on its own broad condition (resample OR fake-hi-res OR a
    # decided lossy verdict) so a green never appears on a file the engine already
    # called fake, independent of the bit-depth prongs' own thresholds.
    bits_regenerated = bool(sp and bd_text.startswith("✓") and (
        sp.resample_detected or sp.fake_hires or sp.verdict_label in ("SUSPICIOUS", "LIKELY_LOSSY")))
    if bits_regenerated:
        m = re.search(r"(\d+)-bit", bd_text)
        depth = f"{m.group(1)}-bit" if m else "Claimed depth"
        bd_text = f"~ {depth} container full, but source depth unverifiable on a transcoded/upsampled stream"
    bd_col = (C.RED if bd_text.startswith("⚠") else C.GREEN if bd_text.startswith("✓")
              else C.ORANGE if bits_regenerated else C.WHITE)
    bd_val = _c(bd_col, bd_text)
    if bits_regenerated:
        bd_val += _c(C.GREY, " — interpolation/requantization regenerates the low-order bits")
    for row in [_kv("Bit-Depth Auth", bd_val), _kv("Header Integrity", auth.header_integrity), _kv("Encoder Trace", _c(C.RED, auth.encoder_trace) if auth.encoder_trace else ""), _kv("MQA", _c(C.ORANGE, "⚠ MQA-encoded — folded payload not verifiable by PCM analysis") if auth.mqa_detected else ""), _kv("Analog Source", _c(C.BLUE, " + ".join(source_flags) + " signature detected") if source_flags else ""), _kv("Side Channel", auth.side_channel_analysis), _kv("Phase Corr.", f"{auth.phase_correlation} {auth.phase_verdict}" if auth.phase_correlation else ""), _kv("Clipping", auth.clipping_verdict if auth.clipping_verdict else ""), _kv("Silence", auth.silence_total_pct)]:
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
    spec = report.spectrogram_path
    if spec and Path(spec).exists():
        print(f"  {_c(C.GREEN,'✓')} Spectrogram → {_c(C.DIM + C.WHITE, str(spec))}")
    else:
        print(f"  {_c(C.RED,'✗')} Spectrogram generation failed (SoX and ffmpeg renderers both errored)")
    if report.analysis_seconds:
        print(f"  {_c(C.GREY, f'Analysed in {report.analysis_seconds:.1f}s')}")
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
    col_w = [34, 6, 12, 10, 5]
    header = f"  {_c(C.GOLD, 'Track'.ljust(col_w[0]))} {_c(C.GOLD, 'DR'.ljust(col_w[1]))} {_c(C.GOLD, 'LUFS'.ljust(col_w[2]))} {_c(C.GOLD, 'NFloor'.ljust(col_w[3]))} {_c(C.GOLD, 'Main'.ljust(col_w[4]))} {_c(C.GOLD, 'Verdict')}"
    print(header); print(_rule("─", W))
    for r in reports:
        name = r.filepath.name[:col_w[0]].ljust(col_w[0])
        dr = _c(_dr_assessment(r.dr_score)[0], r.dr_score.ljust(col_w[1]))
        lufs = r.loudness.lufs_integrated; lufs_s = _c(_lufs_colour(lufs), f"{lufs} LUFS".ljust(col_w[2]) if lufs else "---".ljust(col_w[2]))
        nf = r.loudness.noise_floor_db; nf_s = _c(_noise_colour(nf), f"{nf} dB".ljust(col_w[3]) if nf else "---".ljust(col_w[3]))
        sp = r.authenticity.spectral
        ms_s = _c(_main_score_colour(sp.main_score), str(sp.main_score).ljust(col_w[4])) if sp and sp.verdict_label != "INCONCLUSIVE" else "--".ljust(col_w[4])
        verdict = r.authenticity.spectral_cutoff_verdict or "—"; vshort = verdict[:26]
        print(f"  {_c(C.WHITE, name)} {dr} {lufs_s} {nf_s} {ms_s} {_c(C.DIM + C.WHITE, vshort)}")
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
    # Windows pipes default to cp1252, which cannot encode the report's box-drawing
    # and verdict glyphs — redirecting output would crash with UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try: stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError): pass

    parser = argparse.ArgumentParser(prog="audio_forensic", description="Audio Forensics CLI — comprehensive audio authenticity analysis")
    parser.add_argument("files", nargs="*", help="Audio file(s) to analyse")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--fast", action="store_true", help="Analyse first 60s only")
    parser.add_argument("--info", action="store_true", help="Only show basic metadata")
    parser.add_argument("--workers", type=int, default=0, help="Concurrent files in batch mode (default: auto, up to 3)")
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

    fast = 60.0 if args.fast else None
    workers = args.workers if args.workers > 0 else min(3, os.cpu_count() or 1)
    _Status.begin(len(paths), workers if len(paths) > 1 else 1)
    if len(paths) > 1 and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            reports = list(pool.map(lambda p: build_report(p, fast_secs=fast), paths))
    else:
        reports = [build_report(p, fast_secs=fast) for p in paths]
    _Status.clear()
    if args.json:
        print(json.dumps([_report_to_dict(r) for r in reports], indent=2, default=str))
        return

    for report in reports: print_report(report)
    if len(reports) > 1: print_batch_summary(reports)

if __name__ == "__main__":
    main()
