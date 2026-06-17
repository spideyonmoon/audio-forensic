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
w_full, t_full, fake_full, _ = eng._segment_voting(noise(30.0))
w_wall, t_wall, fake_wall, _ = eng._segment_voting(brickwall(noise(30.0), 15000))
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
print("\n== 15. Codec wall fingerprint ==")
fp = eng._codec_fingerprint(16800)
check("16.8 kHz wall -> MP3 128 kbps fingerprint", fp is not None and "MP3" in fp[0] and "128" in fp[1], f"{fp}")
fp_opus = eng._codec_fingerprint(20440)
check("20.44 kHz wall -> Opus CELT fingerprint", fp_opus is not None and fp_opus[0] == "Opus", f"{fp_opus}")
check("17.9 kHz (between walls) -> no fingerprint", eng._codec_fingerprint(17900) is None)
check("near-Nyquist cutoff -> no fingerprint", eng._codec_fingerprint(21800) is None)

# ---------------------------------------------------------------------------
print("\n== 16. Spliced/partial transcode segmentation ==")
spliced = np.concatenate([noise(15.0), brickwall(noise(15.0), 16800)])
w_sp, t_sp, fake_sp, segs_sp = eng._segment_voting(spliced)
anom = [(t, c, cl) for t, c, cl, _, _ in segs_sp if 0 < c < 0.85 * (SR / 2) and cl > 25.0]
check("spliced file: no majority vote (walls above 16.5k)", not fake_sp, f"{w_sp}/{t_sp} walled")
check("spliced file: >=2 anomalous walled clips found", len(anom) >= 2, f"{len(anom)} anomalous")
check("anomalous clips sit in the transcoded half", all(t >= 14.0 for t, _, _ in anom),
      f"offsets {[f'{t:.1f}s' for t, _, _ in anom]}")
void_anom = [t for t, c, _, vr, pk in segs_sp if 0 < c < 0.85 * (SR / 2) and vr < -110.0 and pk > -40.0]
check("walled clips expose a digital void above the wall", len(void_anom) >= 2 and all(t >= 14.0 for t in void_anom),
      f"{len(void_anom)} void clips")
w_g, t_g, fake_g, segs_g = eng._segment_voting(noise(30.0))
check("genuine: full-band clips, no cliffs", all(c > 21000 and cl < 25.0 for _, c, cl, _, _ in segs_g))
check("genuine clips show no void", all(vr > -105.0 for *_, vr, _ in segs_g),
      f"min void {min(vr for *_, vr, _ in segs_g):.0f} dB")

mute = noise(30.0)
mute[int(6.5 * SR):int(9.5 * SR)] = 0.0
w_m, t_m, fake_m, segs_m = eng._segment_voting(mute)
check("silent clips are skipped, not counted as walled", t_m < 9 and w_m == 0 and not fake_m, f"{w_m}/{t_m} walled")

# ---------------------------------------------------------------------------
print("\n== 17. Sample-rate provenance (upsample / fake hi-res) ==")
SR48 = 48000
eng48 = SpectralEngine(Path("dummy48.flac"), SR48, channels=2)
bins48 = eng48._freq_bins()
n48 = SR48 * 10
white48 = RNG.standard_normal(n48)
X48 = np.fft.rfft(white48)
f48 = np.fft.rfftfreq(n48, 1.0 / SR48)

# Notch + imaging (ffmpeg swr signature): hole at 22.05k, faint energy above it
Xn = X48.copy()
Xn[(f48 > 21850) & (f48 < 22250)] = 0.0
Xn[f48 >= 22250] *= 10 ** (-30 / 20)
fr_n, _, _ = eng48._compute_stft((np.fft.irfft(Xn, n=n48) * 0.3).astype(np.float32))
hit_n = eng48._resample_check(fr_n, bins48)
check("imaging notch at 22,050 Hz -> 44.1k heritage (notch mode)",
      hit_n is not None and hit_n[0] == 44100 and hit_n[1] == "notch", f"{hit_n}")

# Clean wall exactly at the foreign Nyquist (sox-style resampler)
Xw = X48.copy()
Xw[f48 > 22050] = 0.0
fr_w, _, _ = eng48._compute_stft((np.fft.irfft(Xw, n=n48) * 0.3).astype(np.float32))
hit_w = eng48._resample_check(fr_w, bins48)
check("hard wall at exactly 22,050 Hz -> 44.1k heritage (wall mode)",
      hit_w is not None and hit_w[0] == 44100 and hit_w[1] == "wall", f"{hit_w}")

# Aliased mirror (ffmpeg swr weak filter): spectrum above 22.05k = mirrored copy of
# below. Real images carry CONJUGATED coefficients (that's what locks the per-frame
# magnitudes together) — plain copying decorrelates the frames.
Xm = X48.copy()
fold = np.searchsorted(f48, 22050.0)
m_idx = np.arange(1, len(Xm) - fold)
Xm[fold + m_idx] = np.conj(Xm[fold - m_idx]) * 10 ** (-6 / 20)   # image ~6 dB down
xm = np.fft.irfft(Xm, n=n48).real.astype(np.float32) * 0.3
fr_m, _, _ = eng48._compute_stft(xm)
hit_m = eng48._resample_check(fr_m, bins48)
check("aliased mirror around 22,050 Hz -> 44.1k heritage (mirror mode)",
      hit_m is not None and hit_m[0] == 44100 and hit_m[1] == "mirror", f"{hit_m}")

# Native full-band 48k content must stay clean
fr_g, _, _ = eng48._compute_stft((RNG.standard_normal(n48) * 0.3).astype(np.float32))
check("native 48k full-band noise -> no resample flag", eng48._resample_check(fr_g, bins48) is None)

# A codec wall (Opus CELT 20.46k) is NOT a foreign Nyquist — must not fire
Xo = X48.copy()
Xo[f48 > 20460] = 0.0
fr_o, _, _ = eng48._compute_stft((np.fft.irfft(Xo, n=n48) * 0.3).astype(np.float32))
check("codec wall at 20.46 kHz -> not flagged as resample", eng48._resample_check(fr_o, bins48) is None)

# 44.1k container has no lower standard rate to check
check("44.1k container -> never checked", eng._resample_check(f_full, bins) is None)

print("\n== 18. Fake hi-res by insufficient bandwidth ==")
fh = SpectralEngine._is_fake_hires_bandwidth
NYQ96 = 48000.0
# The user's real case: 96k container, content dies at 20k, 35 dB cliff into a
# -84 dB void (above wall-mode's -90, so _resample_check goes blind) -> must fire.
check("96k, 20k wall, -84 dB void -> fake hi-res",
      fh(96000, NYQ96, 20000, 35.0, -84.0, True), "the AAC256->24/96 miss")
# Genuine hi-res: content reaches its own Nyquist -> must NOT fire.
check("96k, content to ~47k -> not fake", not fh(96000, NYQ96, 47000, 1.3, -3.0, True))
# Dark/analog 96k master: low cutoff but audible hiss above (loud void) -> must NOT fire.
check("96k, 20k cutoff but -45 dB hiss above -> not fake (analog)",
      not fh(96000, NYQ96, 20000, 35.0, -45.0, True))
# Gentle mastering rolloff: low cutoff, shallow cliff -> must NOT fire.
check("96k, 20k cutoff but 10 dB gradual rolloff -> not fake",
      not fh(96000, NYQ96, 20000, 10.0, -90.0, True))
# Not hi-res: a 48k container with a 20k wall is normal Redbook-class -> must NOT fire.
check("48k container -> rule does not apply", not fh(48000, 24000.0, 20000, 35.0, -90.0, True))
# Void never measured (cutoff too near Nyquist for a band) -> must NOT fire.
check("void not measured -> not fake", not fh(96000, NYQ96, 20000, 35.0, -120.0, False))
# Boundary: cutoff just under 60% of Nyquist with a wall+void -> fires; just over -> not.
check("cutoff at 58% Nyquist -> fires", fh(96000, NYQ96, int(NYQ96 * 0.58), 30.0, -82.0, True))
check("cutoff at 62% Nyquist -> not fake", not fh(96000, NYQ96, int(NYQ96 * 0.62), 30.0, -82.0, True))

print("\n== 19. Bit-depth forensics (two prongs) ==")
from audio_forensic import _effective_bits, _noise_floor_profile, _bit_depth_verdict

# ---- Prong 1: used-bits / clean-pad detection on MSB-aligned int32 ----
# A genuine 24-bit signal: random 24-bit values placed MSB-aligned (<<8), low bit live.
v24 = (RNG.integers(-(2**23), 2**23, size=200000) | 1).astype(np.int64) << 8
st24 = np.repeat(v24, 2).astype(np.int32)               # duplicate into stereo lanes
check("effective_bits: 24-bit MSB-aligned -> 24", _effective_bits(st24, 2) == 24, str(_effective_bits(st24, 2)))

# Clean 16->24 pad: 16-bit values shifted <<16 (bottom 8 of the 24-bit field dead).
v16 = (RNG.integers(-(2**15), 2**15, size=200000) | 1).astype(np.int64) << 16
st16 = np.repeat(v16, 2).astype(np.int32)
check("effective_bits: clean 16->24 pad -> 16", _effective_bits(st16, 2) == 16, str(_effective_bits(st16, 2)))

# Per-channel: L padded to 16, R full 24 -> must report the deeper channel (24).
mixed = np.empty(v16.size * 2, dtype=np.int32)
mixed[0::2] = v16.astype(np.int32)                      # left: 16-bit pad
mixed[1::2] = v24.astype(np.int32)                      # right: full 24-bit
check("effective_bits: one full channel unmasks depth -> 24", _effective_bits(mixed, 2) == 24,
      str(_effective_bits(mixed, 2)))

# ---- Prong 2: noise-floor profile on synthetic int32 ----
def _int32_from_float(x):
    return np.clip(np.round(x * (2**31)), -(2**31), 2**31 - 1).astype(np.int32)

sr = 48000
nsec = 20
t = np.arange(sr * nsec) / sr
bursts = 0.4 * np.sin(2 * np.pi * 500 * t) * (np.sin(2 * np.pi * 0.15 * t) > 0.6)

# Genuine quiet 24-bit: -110 dBFS white detail in the gaps -> floor well below 16-bit.
deep = bursts + 10 ** (-110 / 20) * RNG.standard_normal(t.size)
pg = _noise_floor_profile(np.repeat(_int32_from_float(deep), 2), 2, sr)
check("floor profile: deep quiet floor < -102 dBFS", pg is not None and pg["floor_db"] < -102.0,
      f"{pg['floor_db']:.1f}" if pg else "None")

# 16-bit dither floor: flat/white noise at ~-95 dBFS in the gaps.
flat16 = bursts + 10 ** (-95 / 20) * RNG.standard_normal(t.size)
pf = _noise_floor_profile(np.repeat(_int32_from_float(flat16), 2), 2, sr)
check("floor profile: 16-bit-level floor is flat", pf is not None and pf["flat"], f"{pf}" if pf else "None")

# Loud master: no exposed floor (quietest window still loud) -> floor > -86.
loud = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * RNG.standard_normal(t.size)
pl = _noise_floor_profile(np.repeat(_int32_from_float(loud), 2), 2, sr)
check("floor profile: loud master floor > -86 dBFS", pl is not None and pl["floor_db"] > -86.0,
      f"{pl['floor_db']:.1f}" if pl else "None")

# Colored (LF-heavy) analog floor: pink-ish noise -> NOT flat.
pink = bursts + 10 ** (-90 / 20) * np.cumsum(RNG.standard_normal(t.size)) / 30.0
pc = _noise_floor_profile(np.repeat(_int32_from_float(pink), 2), 2, sr)
check("floor profile: colored analog floor is not flat", pc is not None and not pc["flat"],
      f"slope={pc['slope_db']:.1f}" if pc else "None")

# ---- Verdict tiers (pure function, no audio) ----
LOUD = {"floor_db": -40.0, "flat": False, "slope_db": -50.0}
DEEP = {"floor_db": -113.0, "flat": True, "slope_db": 0.1}
FLAT16 = {"floor_db": -95.0, "flat": True, "slope_db": -0.2}
ANALOG = {"floor_db": -94.0, "flat": False, "slope_db": -25.0}
check("verdict: clean pad 24<-16 -> upscaled", _bit_depth_verdict(24, 16, LOUD).startswith("⚠ Upscaled"))
check("verdict: 20-in-24 -> reduced-depth tilde", _bit_depth_verdict(24, 20, LOUD).startswith("~ 20 of 24"))
check("verdict: full bits + loud master -> abstain ✓", "not independently confirmable" in _bit_depth_verdict(24, 24, LOUD))
check("verdict: full bits + deep quiet floor -> genuine", _bit_depth_verdict(24, 24, DEEP).startswith("✓ Genuine"))
check("verdict: full bits + flat 16-bit floor -> effective ~16-bit", _bit_depth_verdict(24, 24, FLAT16).startswith("⚠ Effective ~16-bit"))
check("verdict: full bits + colored floor -> analog tilde (no false ⚠)", _bit_depth_verdict(24, 24, ANALOG).startswith("~ Noise floor"))
check("verdict: 16-bit claim + flat 16-bit floor -> consistent ✓", _bit_depth_verdict(16, 16, FLAT16).startswith("✓ 16-bit consistent"))
check("verdict: no profile -> honest unconfirmable", "not independently confirmable" in _bit_depth_verdict(24, 24, None))

print("\n== 20. MDCT quantization-error detector (Derrien JAES 2019) ==")
engM = make_engine("mdct.flac")
# 20a: MDCT/IMDCT-style fold reconstructs via DCT-IV orthonormality (round-trip of a
# single block through the analysis fold + DCT-IV is a real, finite, info-preserving map)
N2 = 2048
win = engM._kbd_window(N2)
# KBD satisfies the Princen-Bradley condition w[n]^2 + w[n+N]^2 == 1 (perfect recon)
half = win[:N2 // 2]
pb = half ** 2 + win[N2 // 2:] ** 2
check("KBD window satisfies Princen-Bradley (w^2+w^2=1)", np.allclose(pb, 1.0, atol=1e-9),
      f"max dev {np.max(np.abs(pb - 1.0)):.2e}")
# 20b: a batch MDCT returns N coefficients for a 2N input
blk = np.random.RandomState(1).randn(3, N2)
coeffs = engM._mdct_batch(blk, win)
check("MDCT maps 2N samples -> N coefficients", coeffs.shape == (3, 1024), f"shape {coeffs.shape}")
# 20c: gamma thresholds are positive and increase with band width K (E~K/12 scaling)
P = 0.01
swb = np.asarray(SpectralEngine._SWB_LONG_44_48, float)
K = np.diff(swb)
mu, sigma = K / 12.0, np.sqrt(K / 180.0)
from scipy.special import ndtr, ndtri
gam = mu + sigma * ndtri(P + (1.0 - P) * ndtr(-mu / sigma))
check("gamma thresholds positive and rise with band width", (gam > 0).all() and gam[-1] > gam[0],
      f"gam[0]={gam[0]:.2f} gam[-1]={gam[-1]:.2f}")
# 20d: genuine wideband white noise (no MDCT lattice) reads LOW (well below the 0.10 wall)
rs = np.random.RandomState(7)
gen = rs.randn(SR * 6).astype(np.float32) * 0.2
Lg = engM._mdct_quant_error(gen, rs.randn(SR * 6).astype(np.float32) * 0.2)
check("genuine white noise -> low MDCT score (< 0.10)", 0 <= Lg < 0.10, f"L={Lg:.3f}")
# 20e: a signal whose MDCT coefficients sit exactly on an integer lattice (a perfect
# synthetic transcode) reads HIGH. Build it directly in the coefficient domain: pick a
# block grid, set integer-scaled coefficients, IMDCT, overlap-add with TDAC.
def imdct_overlap(coeffs_blocks, w, scale):
    N = 1024
    nblk = coeffs_blocks.shape[0]
    out = np.zeros(N * (nblk + 1), dtype=np.float64)
    for i in range(nblk):
        p = _sidct4(coeffs_blocks[i] * scale)                 # folded length-N
        # transpose of the forward fold (PR synthesis) -> windowed 2N frame
        frame = np.empty(N2)
        frame[:N // 2] = p[N // 2:N]
        frame[N // 2:N] = -p[N // 2:N][::-1]
        frame[N:N + N // 2] = -p[:N // 2][::-1]
        frame[N + N // 2:] = -p[:N // 2]
        out[i * N:i * N + N2] += frame * w
    return out
from scipy.fft import idct as _sidct
_sidct4 = lambda c: _sidct(c, type=4, norm='ortho')
rs2 = np.random.RandomState(11)
cb = np.round(rs2.randn(SR * 6 // 1024, 1024) * 6.0)          # integer coefficient lattice
lattice = imdct_overlap(cb, win, scale=80.0).astype(np.float32) / 32768.0
Lt = engM._mdct_quant_error(lattice, None)
check("integer MDCT-coefficient lattice -> higher than genuine noise", Lt > Lg + 0.03,
      f"L_lattice={Lt:.3f} vs L_gen={Lg:.3f}")
# 20f: rate gate — 96k container is out of the swb table's scope -> n/a (-1)
eng96 = SpectralEngine(Path("x96.flac"), 96000, channels=2)
check("non-44.1/48k rate -> detector returns -1 (n/a)",
      eng96._mdct_quant_error(gen, None) == -1.0)

# ---------------------------------------------------------------------------
failed = [n for n, ok, _ in _results if not ok]
print(f"\n{'=' * 60}\n{len(_results) - len(failed)}/{len(_results)} checks passed.")
if failed:
    print("FAILED: " + ", ".join(failed))
    sys.exit(1)
print("All DSP verification checks passed.")
