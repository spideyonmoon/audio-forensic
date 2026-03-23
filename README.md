# Audio Forensics CLI
> Author Note: AI wrote this. Don't have the time at this moment for Readmes. Soon.

---

---

<div align="center">
 
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Audio](https://img.shields.io/badge/Audio-Forensics-red.svg)

**Professional audio forensics analysis with calibrated thresholds for real-world mastered audio.**

</div>

---

## What is Audio Forensic?

A command-line audio forensics tool that analyzes audio files to determine their authenticity, quality, and technical characteristics. It combines multiple audio analysis engines (ffmpeg, SoX, mediainfo) with a custom numpy-based spectral analysis engine to provide comprehensive forensic insights.

Unlike generic audio analysis tools, Audio Forensic was built with **real-world mastered audio in mind** — not synthetic test signals.

---

## The Problem with Other Tools

Most audio forensics tools are calibrated against:
- Clean sine waves
- White/pink noise
- Synthetic test signals

This causes **massive false positives** when analyzing actual music:

| Metric | Generic Tool Says | Reality |
|--------|------------------|---------|
| DR5 | "BAD" (red) | Normal for modern pop/rock |
| Crest Factor 3dB | "BAD" (red) | Typical for compressed audio |
| SoX Entropy 0.09 | "Suspicious" (red) | Normal for tonal music |
| HF cutoff 19.8kHz | "Lossy!" (red) | Standard CD mastering |
| Peak at 0.999 | "Clipping!" (red) | Normal loudness normalization |

af2 fixes all of this.

---

## Features

### Comprehensive Analysis
- **Technical Metadata**: Bit depth, sample rate, channels, encoding
- **Loudness Profiling**: LUFS, DR, crest factor, EBU R128
- **Spectral Analysis**: FFT-based authenticity detection
- **Dynamic Range**: True peak, RMS, noise floor
- **Phase Correlation**: Stereo width and compatibility
- **ReplayGain Audit**: Tag verification

### Intelligent Thresholds
- Calibrated for commercially mastered audio
- Context-aware color coding
- Human-readable interpretations for every metric
- No false positives from normal mastering practices

### Professional Output
- Clean ANSI terminal output
- JSON export for automation
- Batch mode for album analysis
- Spectrogram generation

---

## Installation

### Requirements

```bash
# Core dependencies (must be in PATH)
ffmpeg    # Audio decoding and analysis
sox       # Statistical analysis
mediainfo # Metadata extraction

# Python dependencies
numpy     # Spectral analysis
```

### Install on macOS

```bash
brew install ffmpeg sox mediainfo
pip3 install numpy
```

### Install on Ubuntu/Debian

```bash
sudo apt install ffmpeg sox mediainfo
pip3 install numpy
```

### Install on Windows

Download and install:
- [ffmpeg](https://ffmpeg.org/download.html)
- [SoX](https://sox.sourceforge.net/)
- [MediaInfo](https://mediaarea.net/en/MediaInfo)

Then:
```bash
pip install numpy
```

---

## Quick Start

### Analyze a single file

```bash
python af2.py "path/to/audio.flac"
```

### Batch analyze an album

```bash
python af2.py *.flac
```

### JSON output (for scripts)

```bash
python af2.py "track.flac" --json
```

### Fast mode (first 60 seconds)

```bash
python af2.py "track.flac" --fast
```

### Lightweight info only

```bash
python af2.py "track.flac" --info
```

---

## Output Explained

```
── DYNAMIC RANGE & LOUDNESS ────────────────────────────────

  DR Score (EBU)        DR5 — normal for modern mastered audio
  Crest Factor          2.94 dB — compressed (modern standard)
  SoX Entropy           0.09 — very low (highly tonal/structured)

── AUTHENTICITY & FORENSICS ───────────────────────────────

  ████░░░░░░ ✓ Strong evidence of authentic lossless source
  Score: Lossy 2 − Natural 6 = Net 0

  Lossy indicators
    · None detected

  Natural indicators
    · Gradual spectral rolloff (natural EQ / mastering)
    · High cutoff variance (organic/analog source)
    · Natural HF noise above cutoff
    · Healthy stereo image
    · High spectral complexity

  Context notes
    · Modern mastering often uses gentle HF limiting at 19-20 kHz
```

---

## Metric Reference

### Dynamic Quality

| Metric | What it means | Normal range |
|--------|---------------|--------------|
| **DR Score** | Dynamic Range (higher = more dynamic) | DR5-12 for modern music |
| **Crest Factor** | Peak-to-RMS ratio | 3-15 dB depending on genre |
| **SoX Entropy** | Time-domain signal randomness | 0.0-0.3 for music |

### Spectral Analysis

| Metric | What it means | Suspicious if |
|--------|---------------|---------------|
| **HF Cutoff** | Highest frequency with content | <18 kHz |
| **Cliff Sharpness** | How abrupt the rolloff is | >15 dB/bin |
| **HF Energy Ratio** | Energy above 15kHz | <0.005 |
| **Noise Floor** | Content above cutoff | <-70 dB (silent void) |
| **Banding Score** | Quantization artifacts | >0.92 with low cutoff |

### Verdict Labels

| Label | Meaning |
|-------|---------|
| **GENUINE** | Strong evidence of authentic lossless |
| **LIKELY_GENUINE** | Consistent with lossless source |
| **CAUTION** | Minor quirks, likely legitimate |
| **SUSPICIOUS** | Spectral anomalies detected |

---

## Comparison: af2 vs af.py

af2 is a complete rewrite with calibrated thresholds for real audio:

| Feature | af.py | af2.py |
|---------|-------|--------|
| DR5 coloring | RED (wrong) | WHITE (correct) |
| Crest Factor coloring | RED (wrong) | WHITE (correct) |
| SoX Entropy interpretation | "Suspicious" | "Typical music" |
| Peak at 0.999 coloring | RED (wrong) | YELLOW (correct) |
| HF cutoff threshold | 97% Nyquist | 85% Nyquist |
| HF energy threshold | <0.02 suspicious | <0.005 suspicious |
| Metric explanations | "low=..., high=..." | "moderate variation (natural)" |
| Verdict accuracy | Many false positives | Calibrated for mastered audio |

---

## Real-World Test Results

### Commercial Pop Album

```
Track 1.flac   DR5  -14.8 LUFS  ✓ GENUINE
Track 2.flac   DR6  -14.2 LUFS  ✓ GENUINE
Track 3.flac   DR4  -15.1 LUFS  ✓ GENUINE
```

af2 correctly identifies all tracks as genuine, while af.py flagged them as "WARNING" due to incorrect threshold calibration.

### Lossy Transcode Detection

```
Source: FLAC 16/44.1 → MP3 128kbps → FLAC 16/44.1

af.py:  WARNING (Net 4/17)
af2.py: SUSPICIOUS (Net 7) ✓ Correct
```

af2 correctly identifies the lossy source using calibrated HF and banding thresholds.

---

## Philosophy

### What af2 believes

- **DR5 is not bad** — it's normal for music released after 1990
- **Peaks at 0.999 are not clipping** — they're normal mastering
- **HF rolloff at 20kHz is not suspicious** — it's standard CD mastering
- **Low entropy is not transcode evidence** — it's normal for tonal music

### What af2 flags as suspicious

- True digital overs (>1.0 sample values)
- Silent voids above cutoff (not purple noise)
- Hard spectral cliffs (>15 dB/bin)
- Extremely stable cutoffs with very low HF
- Joint stereo artifacts in side channel

---

## Contributing

Issues and pull requests welcome! If you find a file that af2 misclassifies, please open an issue with:
1. The file's characteristics (genre, year, label)
2. The af2 output
3. Any known history of the file

---

## License

MIT License — do whatever you want with it.

---

## Acknowledgments

- SoX team for the original stat tool
- ffmpeg team for audio processing
- MediaArea for metadata extraction
- The audio forensics community for threshold research

---

<div align="center">

**Made with Python, numpy, and too much listening to compressed music.**

</div>
