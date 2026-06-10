#!/usr/bin/env python3
"""Self-contained verification suite for the advanced DSP forensics in audio_forensic.py.

Validates every mathematical primitive (filtering, correlation, entropy, scatter
collapse, sparsity, voting) on synthetic signals with known ground truth.
No audio files or external tools required. Run:  python test_dsp.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from audio_forensic import (
    SpectralEngine,
    bandpass_filter,
    highpass_filter,
    calculate_autocorrelation,
    calculate_temporal_variance,
)

SR = 44100
RNG = np.random.default_rng(1234)

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))


def noise(seconds: float, amp: float = 0.3) -> np.ndarray:
    return (RNG.standard_normal(int(seconds * SR)) * amp).astype(np.float32)


def brickwall(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Hard FFT brickwall — true digital void above cutoff (a codec wall idealized)."""
    X = np.fft.rfft(x.astype(np.float64))
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    X[f > cutoff_hz] = 0.0
    return np.fft.irfft(X, n=len(x)).astype(np.float32)


def band_energy_db(x: np.ndarray, lo: float, hi: float) -> float:
    X = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    sel = (f >= lo) & (f <= hi)
    return 10 * math.log10(X[sel].mean() + 1e-30)


def make_engine(name: str = "dummy.flac") -> SpectralEngine:
    return SpectralEngine(Path(name), SR, channels=2)


# ---------------------------------------------------------------------------
print("\n== 1. Butterworth filters ==")
x = noise(5.0)
bp = bandpass_filter(x.astype(np.float64), 1000, 2000, SR)
in_band = band_energy_db(bp, 1200, 1800)
below = band_energy_db(bp, 100, 500)
above = band_energy_db(bp, 6000, 10000)
check("bandpass keeps in-band energy", in_band - below > 30, f"in-band {in_band:.1f} dB vs below {below:.1f} dB")
check("bandpass rejects high band", in_band - above > 30, f"in-band {in_band:.1f} dB vs above {above:.1f} dB")

hp = highpass_filter(x.astype(np.float64), 4000, SR)
check("highpass rejects low band", band_energy_db(hp, 5000, 10000) - band_energy_db(hp, 100, 1000) > 30)

# ---------------------------------------------------------------------------
print("\n== 2. Autocorrelation (random vs periodic) ==")
sine = np.sin(2 * np.pi * 441.0 * np.arange(SR) / SR)  # period = 100 samples; lag 50 -> |corr| = 1
check("441 Hz sine -> |autocorr@50| ~ 1", calculate_autocorrelation(sine, lag=50) > 0.95)
check("white noise -> autocorr@50 ~ 0", calculate_autocorrelation(noise(1.0).astype(np.float64), lag=50) < 0.1)

# ---------------------------------------------------------------------------
print("\n== 3. Temporal variance (stable vs modulated energy) ==")
steady = noise(8.0)
tv_steady = calculate_temporal_variance(steady.astype(np.float64), SR)
mod = noise(8.0)
for i in range(0, 8, 2):  # alternate -20 dB every other second
    mod[i * SR:(i + 1) * SR] *= 0.1
tv_mod = calculate_temporal_variance(mod.astype(np.float64), SR)
check("steady noise has low temporal variance", tv_steady < 2.0, f"{tv_steady:.2f} dB")
check("modulated noise has high temporal variance", tv_mod > 5.0, f"{tv_mod:.2f} dB")

# ---------------------------------------------------------------------------
print("\n== 4. Vectorized STFT equals reference frame loop ==")
eng = make_engine()
short = noise(1.0)
mag, phase_hi, hi_start = eng._compute_stft(short)
win = np.hanning(eng.WINDOW)
ref_frames = []
for i in range(0, len(short) - eng.WINDOW, eng.HOP):
    ref_frames.append(np.fft.rfft(short[i:i + eng.WINDOW] * win))
ref = np.array(ref_frames)
check("STFT magnitude matches reference", np.allclose(mag, np.abs(ref), rtol=1e-3, atol=1e-5))
check("STFT high-band phase matches reference", np.allclose(phase_hi, np.angle(ref[:, hi_start:]), rtol=1e-2, atol=1e-2))

# ---------------------------------------------------------------------------
print("\n== 5. Per-frame cutoff detection ==")
bins = eng._freq_bins()
full = noise(10.0)
walled = brickwall(noise(10.0), 16000)
f_full, _, _ = eng._compute_stft(full)
f_wall, _, _ = eng._compute_stft(walled)
co_full = float(np.percentile(eng._cutoff_per_frame(f_full, bins), 95))
co_wall = float(np.percentile(eng._cutoff_per_frame(f_wall, bins), 95))
check("full-band noise cutoff near Nyquist", co_full > 21000, f"{co_full:.0f} Hz")
check("16 kHz brickwall detected", 15500 < co_wall < 16600, f"{co_wall:.0f} Hz")

# ---------------------------------------------------------------------------
print("\n== 6. auCDtect bound frequency (scatter collapse) ==")
avg_full, prob_full, _ = eng._aucdtect_features(f_full, np.zeros((0, 0), dtype=np.float32), bins)
avg_wall, prob_wall, _ = eng._aucdtect_features(f_wall, np.zeros((0, 0), dtype=np.float32), bins)
check("full-band noise: organic scatter to ceiling", avg_full > 19000, f"avg bound {avg_full:.0f} Hz")
check("16 kHz wall: scatter collapse near 16 kHz", 14500 < avg_wall < 18000, f"avg bound {avg_wall:.0f} Hz")
check("mode bound tracks the wall too", prob_wall < 18000, f"mode {prob_wall:.0f} Hz")

# ---------------------------------------------------------------------------
print("\n== 7. High-band phase difference entropy ==")
dummy_frames = np.abs(RNG.standard_normal((50, 200))).astype(np.float32) + 0.1
structured = ((np.arange(50)[:, None] * 0.3 + np.arange(60)[None, :] * 0.05 + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)
random_ph = (RNG.uniform(-np.pi, np.pi, (50, 60))).astype(np.float32)
_, _, ent_struct = eng._aucdtect_features(dummy_frames, structured, bins[:200 + 4])
_, _, ent_rand = eng._aucdtect_features(dummy_frames, random_ph, bins[:200 + 4])
check("structured HF phase -> low entropy", ent_struct < 2.0, f"{ent_struct:.2f} bits")
check("randomized HF phase -> high entropy", ent_rand > 4.5, f"{ent_rand:.2f} bits (max {math.log2(36):.2f})")

# ---------------------------------------------------------------------------
print("\n== 8. Segment voting (AFD PRO) ==")
w_full, t_full, fake_full = eng._segment_voting(noise(30.0))
w_wall, t_wall, fake_wall = eng._segment_voting(brickwall(noise(30.0), 15000))
check("full-band noise passes the vote", (not fake_full) and w_full == 0, f"{w_full}/{t_full} walled")
check("15 kHz walled file fails the vote", fake_wall and w_wall == t_wall, f"{w_wall}/{t_wall} walled")

# ---------------------------------------------------------------------------
print("\n== 9. Spectral sparsity (psychoacoustic holes) ==")
sp_frames = np.ones((20, len(bins)), dtype=np.float32)
cut_idx = int(20000 / (bins[1] - bins[0]))
hole_mask = RNG.uniform(size=(20, cut_idx)) < 0.4
region = sp_frames[:, :cut_idx]
region[hole_mask] = 1e-7  # -140 dB holes
sparsity = eng._spectral_sparsity(sp_frames, bins, 20000)
check("40% zeroed bins -> sparsity ~ 0.4", 0.3 < sparsity < 0.5, f"{sparsity:.3f}")
check("dense spectrum -> sparsity ~ 0", eng._spectral_sparsity(np.ones((20, len(bins)), dtype=np.float32), bins, 20000) < 0.01)

# ---------------------------------------------------------------------------
print("\n== 10. Ultrasonic envelope correlation ==")
env = np.abs(RNG.standard_normal(100)).astype(np.float32) + 0.5
corr_frames = np.full((100, len(bins)), 1e-6, dtype=np.float32)
bh = bins[1] - bins[0]
corr_frames[:, int(1000 / bh):int(8000 / bh)] = env[:, None]
corr_frames[:, int(16000 / bh):int(22000 / bh)] = env[:, None] * 0.001  # HF breathes with music
c_genuine = eng._ultrasonic_envelope_correlation(corr_frames, bins)
corr_frames[:, int(16000 / bh):int(22000 / bh)] = (np.abs(RNG.standard_normal(100)).astype(np.float32) + 0.5)[:, None] * 0.001
c_fake = eng._ultrasonic_envelope_correlation(corr_frames, bins)
check("coupled HF envelope -> corr ~ 1", c_genuine > 0.9, f"{c_genuine:.2f}")
check("independent injected HF -> corr ~ 0", abs(c_fake) < 0.3, f"{c_fake:.2f}")

# ---------------------------------------------------------------------------
print("\n== 11. Silence / dither / vinyl analyser ==")
# Case A: clean silence inside genuine music -> measured clean, NO credit from the
# function itself (lossy encoders zero silence too; the conditional credit lives in analyse())
music = noise(45.0, amp=0.3)
music[5 * SR:8 * SR] = 0.0
s, rs, ratio, v, _ = eng._silence_and_vinyl(music, 22000)
check("clean digital silence -> measured, no auto-credit", s == 0 and 0 <= ratio < 0.15, f"score {s}, ratio {ratio:.3f}")

# Case B: codec hash in silence (HF noise present in 'silent' passage) -> +50 penalty
music_lp = brickwall(noise(45.0, amp=0.3), 15000)   # music has ~no 16k+ energy
hash_noise = noise(45.0, amp=0.0008)                # faint full-band codec hash (well below -40 dB silence gate)
fake = music_lp + hash_noise
fake[5 * SR:8 * SR] = hash_noise[5 * SR:8 * SR] * 1.5  # 'silence' still carries the hash
s2, _, ratio2, _, _ = eng._silence_and_vinyl(fake, 15000)
check("codec hash in silence -> +50 penalty", s2 == 50 and ratio2 > 0.3, f"score {s2}, ratio {ratio2:.3f}")

# Case C: vinyl — random stable hiss above the musical cutoff -> -40 credit
vinyl = brickwall(noise(60.0, amp=0.3), 16000) + noise(60.0, amp=0.003)
s3, _, _, v3, _ = eng._silence_and_vinyl(vinyl, 16000)
check("analog hiss above cutoff -> vinyl detected", v3 and s3 <= -40, f"score {s3}, vinyl {v3}")

# Case D: digital void above cutoff -> +20 upscale suspicion
void = brickwall(noise(60.0, amp=0.3), 16000)
s4, _, _, v4, _ = eng._silence_and_vinyl(void, 16000)
check("digital void above cutoff -> +20 penalty", s4 == 20 and not v4, f"score {s4}")

# ---------------------------------------------------------------------------
print("\n== 12. Cassette source profiler (Rule 11) ==")
tape = brickwall(noise(60.0, amp=0.3), 14000) + noise(60.0, amp=0.01)  # music + audible hiss
tape_frames, _, _ = eng._compute_stft(tape)
cs, cev, hiss = eng._cassette_source(tape, tape_frames, bins, 14000.0, 150.0, mp3_detected=False)
check("tape hiss + flutter -> cassette score >= 30 with hiss", cs >= 30 and hiss, f"score {cs}: {len(cev)} rules hit")
sterile = brickwall(noise(60.0, amp=0.3), 14000)
sterile_frames, _, _ = eng._compute_stft(sterile)
cs_clean, _, hiss_clean = eng._cassette_source(sterile, sterile_frames, bins, 14000.0, 5.0, mp3_detected=True)
check("sterile walled file -> no hiss, no veto", (cs_clean < 30 or not hiss_clean), f"score {cs_clean}, hiss {hiss_clean}")

# ---------------------------------------------------------------------------
print("\n== 13. Psychoacoustic artefacts (smoke + sanity) ==")
ps, pev, pre, alias, comb = eng._psychoacoustic_artifacts(noise(20.0), f_full, bins, 16000.0, False)
check("clean noise: no pre-echo / aliasing / comb flags", ps == 0 and not comb, f"score {ps}, pre {pre:.1f}%, alias {alias:.2f}")
check("aliasing corr near zero on uncorrelated bands", alias < 0.3, f"{alias:.2f}")

# ---------------------------------------------------------------------------
print("\n== 14. Header integrity (Fakin' the Funk) ==")
fd, tmp = tempfile.mkstemp(suffix=".mp3")
os.write(fd, b"\x00" * 1_000_000)  # 1 MB payload
os.close(fd)
try:
    e2 = SpectralEngine(Path(tmp), SR, channels=2, claimed_duration=10.0, claimed_bitrate_kbps=320)
    dmm, bmm, reasons = e2._check_header_integrity(decoded_duration=9.0)
    check("1s duration gap -> duration mismatch", dmm)
    check("1MB/10s vs 320kbps claim -> bitrate mismatch", bmm, f"actual ~{1_000_000 * 8 / 10 / 1000:.0f} kbps")
    e3 = SpectralEngine(Path(tmp), SR, channels=2, claimed_duration=25.0, claimed_bitrate_kbps=320)
    dmm2, bmm2, _ = e3._check_header_integrity(decoded_duration=25.1)
    check("honest header passes", not dmm2 and not bmm2)
finally:
    os.unlink(tmp)

# ---------------------------------------------------------------------------
failed = [n for n, ok, _ in _results if not ok]
print(f"\n{'=' * 60}\n{len(_results) - len(failed)}/{len(_results)} checks passed.")
if failed:
    print("FAILED: " + ", ".join(failed))
    sys.exit(1)
print("All DSP verification checks passed.")
