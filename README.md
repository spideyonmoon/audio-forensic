# Audio Forensic

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Audio](https://img.shields.io/badge/Audio-Forensics-red.svg)

**State-of-the-art lossless-authenticity forensics for real-world music.**

Detects fake lossless files (lossy transcodes hiding in FLAC/ALAC/WAV containers)
with an 11-rule DSP forensic engine, measured codec fingerprints, and a unified
0–100 Main Score. Detection is heuristic: genuine mastering can resemble codec artifacts.

</div>

---

## What it does

You hand it audio files. It tells you, with evidence, whether they are what they
claim to be:

- **Transcode detection** — was this FLAC once an MP3, AAC, Opus, or Vorbis file?
- **Codec fingerprinting** — *which* encoder and bitrate left the wall (measured
  LAME/AAC/Vorbis lowpass frequencies, Opus' CELT 20 kHz band limit)
- **Spliced/partial transcode detection** — which time regions are walled, reported as timestamps
- **Fake hi-res detection** — was this "24/48" upsampled from a 44.1 kHz CD? Resamplers
  leave a wall, a notch, or an aliased mirror at the source Nyquist; all three are caught
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
python audio_forensic.py ./album_dir           # scan a directory for audio files
python audio_forensic.py a.flac b.mp3 --compare # rank variants of one track, pick the best
python audio_forensic.py track.flac --json     # machine-readable output
python audio_forensic.py track.flac --fast     # first 60 s only
python audio_forensic.py track.flac --info     # metadata only, no DSP
python audio_forensic.py *.flac --workers 4    # batch concurrency (default: auto, ≤3)
```

Analysis time depends on track length, CPU, and enabled transform searches. A live status line shows the current stage,
a progress bar, and a self-calibrating ETA.

## The Main Score (0–100)

Every detector feeds one heuristic evidence score. Higher scores indicate stronger
transcode indicators; the score is not a calibrated probability or proof of provenance.

| Score | Verdict | Meaning |
|-------|---------|---------|
| 0–10 | **GENUINE** | No strong lossy indicators detected; source history unverified |
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
| AAC 256–320 kbps (full bandwidth, no wall) | **55 SUSPICIOUS** (MDCT quantization-error detector) |
| Opus 64–192 kbps | **88 LIKELY_LOSSY** (every bitrate — CELT 20 kHz fingerprint) |
| Vorbis q2–q4 | **91–100 LIKELY_LOSSY** |
| 24/96 master → MP3 320 → 24-bit ALAC | **88 LIKELY_LOSSY** |
| 16/44.1 upsampled to "24/48" or "24/96" (fake hi-res) | **45–75 SUSPICIOUS** — named as a sample-rate counterfeit |
| Half-genuine / half-MP3 splice | **45 CAUTION** + walled regions listed with timestamps |
| Genuine / dark master / vinyl / cassette / mono controls | **0 GENUINE** (zero false positives) |

High-bitrate AAC (256/320 kbps) keeps full bandwidth and leaves no lowpass wall, so
the cutoff-based detectors are blind to it. These are now caught by a **MDCT
quantization-error detector** (Derrien, JAES 2019): an AAC encoder rounds scaled MDCT
coefficients to integers, and that rounding is *idempotent* — re-running the same MDCT
on the decoded "lossless" PCM reproduces near-zero error across many scalefactor bands,
a possible codec fingerprint. Silent channels, inactive anchors, and unpopulated
bands are excluded: zeros also re-round perfectly in genuine audio. This is a blind
heuristic, not a guarantee of zero false positives.

Known limits: HE-AAC/AAC+ (SBR) and AAC with TNS regenerate or reshape the high band, so
no MDCT grid lines up — they evade the AAC quant-error test. A separate Vorbis-window
search now detects persistent transform-grid zeros in controlled q4/q6/q8/q10
stereo fixtures at 44.1/48 kHz, including conversion to 24-bit FLAC. It currently
models common 2048/256-sample blocks and allows long-block alignment to shift
after short blocks. Transition windows themselves, other block sizes,
post-processing, noise/dither, and resampling can hide the trace. A
negative result does not exclude Vorbis or Opus ancestry. Ogg is a container;
this new test targets Vorbis, not every codec Ogg can carry.

## How it works — the forensic suite

One ffmpeg decode feeds a cached STFT plus targeted MDCT searches. Highlights:

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
- **Sample-rate provenance** — a "24/48" file upsampled from CD carries a fingerprint
  at exactly 22,050 Hz: a hard wall (clean resampler), a deep notch with imaging noise
  above it, or — for weak anti-imaging filters like ffmpeg's default — an aliased
  *mirror image* of the sub-Nyquist spectrum, exposed by per-frame mirror correlation.
  Nothing natural has features at precisely a foreign Nyquist. Verdict reads
  "Sample-rate counterfeit — upsampled from 44.1 kHz (fake hi-res)".
- **auCDtect-style bound frequency** — spectral scatter collapse exposes the
  statistical void a codec leaves even when noise is pasted on top.
- **Silence dither analysis** — codec hash inside "silent" passages (+50). Asymmetric
  by design: clean silence is only weak evidence, because lossy encoders zero out
  silence too.
- **Psychoacoustic artifacts** — pre-echo (MDCT smearing), filterbank aliasing
  correlation, the MP3 32-band 689 Hz subband comb.
- **MDCT quantization-error lattice** — the high-bitrate AAC backstop. Re-applies the
  AAC analysis MDCT to the decoded PCM; if scaled coefficients re-round to integers
  across many scalefactor bands, an AAC quantizer's fingerprint is baked into the
  "lossless" file even though it kept full bandwidth with no wall to betray it.
- **Vorbis transform grid** — tests reconstructed L/R channels with the
  [Vorbis I window](https://www.xiph.org/vorbis/doc/Vorbis_I_spec.html), searching
  every sample alignment over up to 12 clips within the first 180 seconds. A
  repeated excess of near-zero coefficients on alignments sharing a 128-sample
  short-hop grid, compared with equally searched off-grid transforms, sets a
  minimum score of 55. This handles long-block phase shifts after transient-driven
  short blocks; stationary-noise fixtures alone missed that requirement. At least four clips and
  half the active clips must support the pattern. Background-relative gates
  reject tested tones and mastering lowpasses. This is provisional transform-codec
  evidence, not a unique encoder identification or a measured probability.
- **Anti-forensics exposure** — fake ultrasonic noise injected above a codec wall is
  caught by envelope-correlation + scatter-collapse cross-checks.
- **Analog vetoes** — vinyl (random, stable hiss + click transients) and cassette
  (tape hiss + natural slope + wow/flutter) subtract evidence instead of adding it;
  a real tape rip with a 14 kHz ceiling is *not* a transcode.
- **Source integrity** — two-prong bit-depth forensics: a per-channel *used-bits*
  pass that proves clean integer padding (16-in-24), plus a *noise-floor* pass that
  reads the effective dynamic range. When a quiet passage exposes the floor it can
  confirm genuine >16-bit content, or flag a flat 16-bit dither floor hiding under a
  24-bit container — while honestly abstaining on loud masters where the floor is
  masked (a transparent 16→24 upsample of a noisy source is physically
  indistinguishable from native 24-bit). Plus header duration/bitrate plausibility
  and lossy-encoder fingerprints left in tags.

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

See [accuracy change notes](ACCURACY_NOTES.md) for detector design, regression
results, real-file observations, and remaining limits.

```bash
python -X utf8 test_dsp.py   # 79 synthetic DSP checks, no audio files needed
python -X utf8 -m unittest -v test_accuracy   # accuracy regressions + optional FFmpeg encodes
python -X utf8 -m unittest -v test_vorbis    # Vorbis-to-24-bit-FLAC matrix, transient switching + genuine controls
```

The accuracy regressions cover dual mono, quiet side channels, tones, silence,
transients, and AAC 256/320 kbps at 44.1/48 kHz. The AAC fixtures disable TNS;
they verify this detector's supported case. These controlled checks do not measure
precision or recall on real music. Corpus-level accuracy still needs independently
sourced lossless masters and matched transcodes, including difficult negative controls.

## License

MIT — do whatever you want with it.
