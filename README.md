# Audio Forensic

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Audio](https://img.shields.io/badge/Audio-Forensics-red.svg)

**State-of-the-art lossless-authenticity forensics for real-world music.**

Detects fake lossless files (lossy transcodes hiding in FLAC/ALAC/WAV containers)
with an 11-rule DSP forensic engine, measured codec fingerprints, and a unified
0–100 Main Score — calibrated so genuine mastered audio never gets falsely flagged.

</div>

---

## What it does

You hand it audio files. It tells you, with evidence, whether they are what they
claim to be:

- **Transcode detection** — was this FLAC once an MP3, AAC, Opus, or Vorbis file?
- **Codec fingerprinting** — *which* encoder and bitrate left the wall (measured
  LAME/AAC/Vorbis lowpass frequencies, Opus' CELT 20 kHz band limit)
- **Spliced/partial transcode detection** — which time regions are walled, reported as timestamps
- **Bit-depth authenticity** — 24-bit container, but do all 24 bits carry signal?
- **Header forensics** — forged duration/bitrate headers ("Fakin' the Funk" checks)
- **Analog source profiling** — vinyl surface noise and cassette tape signatures are
  recognized and *excused*, not flagged
- **Full loudness/dynamics report** — LUFS, DR, true peak, crest factor, streaming
  normalization deltas, ReplayGain audit, phase correlation, clipping, silence map
- **Spectrogram** — SoX-rendered PNG next to every analyzed file

## Quick start

```bash
# Requirements in PATH: ffmpeg, sox, mediainfo
pip install -r requirements.txt   # numpy + scipy

python audio_forensic.py "track.flac"          # full forensic report
python audio_forensic.py *.flac                # batch an album (live ETA, summary table)
python audio_forensic.py track.flac --json     # machine-readable output
python audio_forensic.py track.flac --fast     # first 60 s only
python audio_forensic.py track.flac --info     # metadata only, no DSP
python audio_forensic.py *.flac --workers 4    # batch concurrency (default: auto, ≤3)
```

A 4-minute FLAC analyzes in ~3–5 s. A live status line shows the current stage,
a progress bar, and a self-calibrating ETA.

## The Main Score (0–100)

Every detector feeds one number. 0 = pristine lossless, 100 = certain transcode.

| Score | Verdict | Meaning |
|-------|---------|---------|
| 0–10 | **GENUINE** | Strong evidence of authentic lossless source |
| 11–30 | **LIKELY_GENUINE** | Consistent with genuine lossless |
| 31–54 | **CAUTION** | Minor spectral quirks — possibly legitimate |
| 55–85 | **SUSPICIOUS** | Strong lossy indicators — probable transcode |
| 86–100 | **LIKELY_LOSSY** | Fake lossless, high certainty |

## Measured detection performance

Synthetic pink-noise fixtures, encode → decode → FLAC/ALAC (i.e. fake lossless), at
both 44.1 and 48 kHz:

| Source | Result |
|--------|--------|
| MP3 64–320 kbps | **88–100 LIKELY_LOSSY** (all bitrates, both sample rates) |
| AAC 96–192 kbps | **88–100 LIKELY_LOSSY** |
| Opus 64–192 kbps | **88 LIKELY_LOSSY** (every bitrate — CELT 20 kHz fingerprint) |
| Vorbis q2–q4 | **91–100 LIKELY_LOSSY** |
| 24/96 master → MP3 320 → 24-bit ALAC | **88 LIKELY_LOSSY** |
| Half-genuine / half-MP3 splice | **45 CAUTION** + walled regions listed with timestamps |
| Genuine / dark master / vinyl / cassette / mono controls | **0 GENUINE** (zero false positives) |

Known limits: AAC ≥256 kbps and Vorbis q6+ encode pink noise at full bandwidth and
are spectrally invisible on synthetic fixtures — real music leaves more artifacts,
but treat "transparent-bitrate AAC" as detectable only sometimes.

## How it works — the forensic suite

One ffmpeg decode and one cached STFT feed every detector. Highlights:

- **Segment voting (9 clips)** — 2 s clips spread across the file, each checked for a
  frequency wall; majority = whole-file lossy ancestry (+55). The wall threshold is
  *adaptive*: a >30 dB cliff backed by a verified digital void (or a codec-fingerprint
  match) moves the wall up to the cutoff, which is what catches 320 kbps walls at 20.5 kHz.
- **Codec wall fingerprints** — the cutoff is compared against *measured* encoder
  lowpass tables (LAME per bitrate at 44.1 **and** 48 kHz, ffmpeg-AAC, Vorbis, Opus'
  bitrate-independent 20.46 kHz CELT limit). Published spec tables are wrong; these
  were measured from real encodes. A hit gated on void/cliff evidence adds +10 and
  names the encoder in the report.
- **Spliced/partial detection** — per-clip cutoffs + per-clip cliff depth; ≥2 walled
  clips in an otherwise full-band file report exact mm:ss regions (+30–55 by coverage
  and fingerprint).
- **auCDtect-style bound frequency** — spectral scatter collapse exposes the
  statistical void a codec leaves even when noise is pasted on top.
- **Silence dither analysis** — codec hash inside "silent" passages (+50). Asymmetric
  by design: clean silence is only weak evidence, because lossy encoders zero out
  silence too.
- **Psychoacoustic artifacts** — pre-echo (MDCT smearing), filterbank aliasing
  correlation, the MP3 32-band 689 Hz subband comb.
- **Anti-forensics exposure** — fake ultrasonic noise injected above a codec wall is
  caught by envelope-correlation + scatter-collapse cross-checks.
- **Analog vetoes** — vinyl (random, stable hiss + click transients) and cassette
  (tape hiss + natural slope + wow/flutter) subtract evidence instead of adding it;
  a real tape rip with a 14 kHz ceiling is *not* a transcode.
- **Source integrity** — effective-bit-depth probe (16-in-24 padding detection),
  header duration/bitrate plausibility, lossy-encoder fingerprints left in tags.

Every fired rule prints a human-readable evidence line, so the verdict is auditable.

## Calibrated for real music

Generic tools flag normal mastering as suspicious. This one does not:

| Trait | Generic tool | Audio Forensic |
|-------|--------------|----------------|
| DR5, crest 3 dB | "BAD" | Normal modern mastering |
| Peak at 0.999 | "Clipping!" | Normal limiting |
| 19–20 kHz mastering LPF | "Lossy!" | Legitimate unless a *measured codec wall* + void backs it |
| Vinyl/tape HF rolloff | "Lossy!" | Analog signature → evidence subtracted |
| Dark/quiet masters | "Suspicious" | 0 on the control fixtures |

## Output

Full ANSI-colored terminal report: identity/tags (every tag the file carries),
technical specs, loudness graph + EBU R128 + streaming deltas, dynamics, the
forensic verdict with evidence lists, SoX acoustic measurements, and per-file
timing. `--json` emits the entire structure for scripting. Batch mode adds an
album summary table with DR/LUFS/score per track and DR-outlier warnings.

## Verification

```bash
python test_dsp.py   # 42 self-contained synthetic-signal checks, no audio files needed
```

## License

MIT — do whatever you want with it.
