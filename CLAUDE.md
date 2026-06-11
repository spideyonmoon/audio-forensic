# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Audio Forensic** is a command-line tool for comprehensive audio authenticity analysis. Unlike generic audio tools that falsely flag normal mastered audio, this tool uses calibrated thresholds specifically tuned for real-world commercially mastered audio.

- **Language**: Python 3.8+
- **Main file**: `audio_forensic.py` (~1,900 lines)
- **Entry point**: `main()` function using argparse for CLI
- **Key dependencies**: numpy (base spectral engine), scipy (advanced forensic suite — degrades gracefully if missing), plus external tools (ffmpeg, sox, mediainfo)
- **Tests**: `test_dsp.py` — self-contained synthetic-signal verification of every DSP metric (`python test_dsp.py`, no audio files/tools needed)

## Development & Running

### Install dependencies
```bash
pip install -r requirements.txt
# Also install system tools:
# - ffmpeg (audio decoding)
# - sox (statistical analysis)
# - mediainfo (metadata extraction)
```

### Run the tool
```bash
# Single file analysis (full forensics)
python audio_forensic.py "path/to/audio.flac"

# Batch processing (multiple files)
python audio_forensic.py *.flac

# JSON output (for scripting)
python audio_forensic.py track.flac --json

# Fast mode (first 60 seconds only)
python audio_forensic.py track.flac --fast

# Lightweight info only (no spectral analysis)
python audio_forensic.py track.flac --info

# Control batch concurrency (default: auto, up to 3 files at once)
python audio_forensic.py *.flac --workers 4
```

## Architecture

### Data Models (Dataclasses)

Located at the top of `audio_forensic.py`:
- **`AudioTags`**: Metadata (title, artist, album, ReplayGain tags)
- **`AudioTechnical`**: Technical specs (bit depth, channels, sample rate, duration)
- **`LoudnessProfile`**: All loudness metrics from ffmpeg/sox (LUFS, DR, crest factor, RMS, etc.)
- **`SpectralAnalysis`**: FFT-based verdict engine output (cutoff frequency, scoring, evidence)
- **`AuthenticityReport`**: Final combined verdict (includes phase, clipping, bit depth checks)
- **`ForensicReport`**: Complete report wrapping all the above plus file path and spectrogram

### Key Components

#### 1. **Tool Extractors** (`extract_*` functions)
Extract data from external tools and parse their output:
- `extract_mediainfo()` → AudioTags & AudioTechnical (from mediainfo JSON)
- `extract_sox_stats()` → dict of SoX statistics
- `extract_loudness()` → (LoudnessProfile, DR score) — **one** ffmpeg process: the stream is split through astats + ebur128 + drmeter simultaneously via `-filter_complex asplit`
- `check_bit_depth_authenticity()` → 24-bit genuine vs. padded-16-bit detection
- `audit_replaygain()` → ReplayGain tag validation
- `generate_spectrogram()` → PNG via SoX (chosen for visual quality), ffmpeg fallback. **`-y` must stay 2ⁿ+1 (513)** — SoX maps that to a fast DFT size; 512 triggers a resampling path ~20× slower.

**Byproduct metrics** (numpy, computed from the SpectralEngine's decode — no extra ffmpeg processes; the old `aphasemeter`/`astats=clipping` invocations were silently-broken filter syntax):
- `measure_phase_correlation(mid, side, sr)` → per-100ms L/R Pearson correlation
- `detect_clipping(mid, side)` → samples at 16-bit full scale
- `map_silence(mid, sr, duration)` → silent runs < -60 dBFS for ≥ 0.5 s
- `_noise_floor_from_audio(mid, sr)` → fallback when astats reports nan/inf (5th pct of per-100ms RMS)

**Metadata forensics**:
- `extract_mediainfo` surfaces EVERY tag: known fields go to AudioTags, everything else into `tags.other` (philosophy: show all the data a file carries)
- `check_bit_depth_authenticity(path, claimed, duration)` → trailing-zero analysis of a 30 s mid-track s32 decode; measures *effective* bits (16-in-24 padding leaves the bottom 8 bits dead in every sample; a bit rank must be hit by ≥0.01% of samples to count)
- `detect_encoder_trace(tags, tech, path)` → lossy-encoder fingerprints (LAME/Fraunhofer/"320kbps"/joint stereo) inside a lossless container's metadata; display-only red flag, not scored

**Important**: For formats SoX can't read (MP3, M4A, AAC, OGG, OPUS, WMA, APE), `extract_sox_stats` pipes a WAV decode from ffmpeg straight into SoX's stdin — no temp files anywhere in the pipeline.

#### 2. **SpectralEngine** (NumPy/SciPy DSP forensic engine)
Core authenticity verdict engine. One stereo decode feeds everything (mid `(L+R)/2` for mono detectors, side `(L−R)/2` for joint-stereo forensics). One vectorized, chunked complex STFT (`_compute_stft()`) is cached and reused by every detector — magnitude everywhere, phase kept only ≥10 kHz for auCDtect. Silent frames are masked out of all statistics (`_active_frame_mask()`).

**Key constants**:
- `WINDOW = 4096; HOP = 2048` — FFT frame size and stride
- `CUTOFF_DB = -65.0` — Energy threshold for detecting spectral rolloff
- `NYQUIST_MARGIN = 0.85` — Cutoff must be below 85% of Nyquist to be suspicious
- `TIME_DOMAIN_CAP_S = 180` — envelope/filterbank analyses capped to bound CPU/RAM
- `MP3_CUTOFFS` — empirically measured LAME lowpass cutoffs per bitrate (-65 dB point); still used for MP3-comb gating
- `CODEC_WALLS` — measured codec lowpass walls as `(codec, profile, hz, tol)` tuples covering LAME (44.1k AND 48k — walls shift with sample rate), ffmpeg-AAC, Vorbis, and Opus' bitrate-independent CELT 20,460 Hz limit. Measured via `testdata/make_fixtures.py` + `testdata/measure.py`; NEVER replace with published spec values (they're wrong). `_codec_fingerprint(cutoff_hz)` returns the nearest in-tolerance entry (None at ≥98% Nyquist).

**Base scoring** (legacy evidence engine, feeds 45 of the 100 main-score points):
- `SCORE_*` / `NATURAL_*` constants, `MAX_LOSSY_SCORE = 14`

**Base spectral methods**:
- `_decode_audio()` / `_decode_stereo()` — FFmpeg to numpy array conversion
- `_cutoff_per_frame()` — vectorized per-frame spectral rolloff detection
- `_sharpness()` / `_cliff_depth()` — cliff gradient (dB/bin) and drop across ±400 Hz (codec walls fall 35+ dB/800 Hz; natural fades don't)
- `_hf_energy_ratio()`, `_banding_score()`, `_noise_floor_above_cutoff()`, `_side_channel_anomaly()`, `_lpf_scan()`, `_dsd_scan()`, `_spectral_entropy()`

**Advanced forensic suite** (scipy; each method returns score deltas + evidence strings):
- `_check_header_integrity()` — Fakin' the Funk: container duration vs decoded sample count (free — uses the existing decode), bitrate-vs-filesize plausibility for lossy containers
- `_segment_voting()` — AFD PRO: 9 spread 2s clips wall-checked, majority vote (+55); returns per-clip `(offset_s, cutoff_hz, cliff_db)` and skips silent clips (their cutoff reads 0 and would poison the vote). Wall threshold is **adaptive**: a >30 dB cliff with a verified digital void above it **or a codec-fingerprint match** moves the wall from 16.5 kHz to cutoff+400. The fingerprint path matters: AAC's own residue above its cutoff defeats the void check (e.g. AAC 192 @48k scored 36 until fingerprint-arming pushed it to 100)
- **Partial/spliced transcode rule** (in `analyse()`, `elif` of the majority vote): ≥2 non-majority clips with cutoff < min(global−2k, 0.85·Nyquist) AND per-clip cliff >25 dB → +30 (+40 if ≥4 clips, +15 more if their median cutoff fingerprints) with mm:ss regions in `result.segment_map`, rendered under the Segment Vote row
- **Codec wall fingerprint rule** — `_codec_fingerprint(cutoff_hz)` hit gated on (void verified OR cliff >30 dB) and no analog veto → +10, names encoder+bitrate in evidence. This is what lifts high-bitrate walls (MP3 192–320, Opus, Vorbis q4) from 78/SUSPICIOUS to 88+/LIKELY_LOSSY while arbitrary mastering LPFs (not on a measured frequency) stay clear
- `_resample_check(frames, bins)` — **sample-rate provenance / fake hi-res** (numpy-only, runs even without scipy). Checks each `RESAMPLE_SOURCE_RATES` Nyquist below the container's for three resampler signatures: **"wall"** (spectrum ends in a cliff exactly at the foreign Nyquist, ≥40 dB into a <−90 dB void), **"notch"** (≥20 dB hole at exactly fn with imaging/injected energy above — the energy above pushes the measured cutoff to full bandwidth and defeats every cutoff-based detector), and **"mirror"** (weak anti-imaging filters like ffmpeg swr default pass aliased images almost unattenuated: the spectrum above fn is a mirror copy of below; per-frame magnitude correlation with **fractional-bin interpolation** and NO smoothing — images carry conjugated coefficients so frame magnitudes mirror exactly per bin; smoothing mixes in folded k≥2 images and dilutes it. Measured: swr fixture 0.49, genuine noise 0.003, threshold 0.35). Any hit → +45, `_verdict` overrides anything below LIKELY_LOSSY to SUSPICIOUS "Sample-rate counterfeit — upsampled from X kHz (fake hi-res)", adds a caveat that bit-depth verification can't see through resampling (interpolation regenerates low-order bits — a "verified 24-bit" reading proves nothing on an upsample). A wall sitting on a foreign Nyquist does NOT arm the adaptive segment vote (it's resample, not codec, evidence), and clean-silence credit is withheld when resample fires
- `_is_fake_hires_bandwidth(sample_rate, nyquist, cutoff_hz, cliff_depth, void_db, void_measured)` — **fake hi-res by insufficient bandwidth** (static, scipy block — needs the void measurement). The backstop for when `_resample_check` goes blind: a Redbook→lossy→hi-res chain (e.g. **Redbook → AAC 256 → 24/96**) where the codec lowpass sits *below* the foreign Nyquist, killing the energy near it — so notch/mirror need a live "edge" they don't have, and wall-mode needs a <−90 dB void but the codec leaves only ~−85. All three resample modes miss; the file shows `Resample Check ✓` and a green "Verified 24-bit". This rule fires on the container itself: `sample_rate ≥ 88200` AND `cutoff < 0.6·Nyquist` AND `cliff > 25 dB` AND measured `void < −80 dB` → +20, sets `result.fake_hires`, and `_verdict` forces ≥SUSPICIOUS "Fake hi-res — N kHz container but bandwidth ends at M kHz (upsampled)". Gated hard so genuine high-rate masters (content to Nyquist) and analog/gentle rolloffs (shallow cliff or audible hiss above) never trip it. Only runs when `_resample_check` did NOT already fire (no double-count). Boundaries unit-tested in test_dsp §18
- `_aucdtect_features()` — bound frequency via spectral scatter collapse (5-bin sliding std of log power; bins <-110 dB rel clamped so decoder residue reads as void) + high-band phase-difference entropy (>4.5 bits with depressed cutoff = quantized HF phase). Note: the bound check is defeated by 16-bit re-quantization noise on 16-bit fakes — the void/wall detectors cover those; auCDtect covers float/24-bit fakes
- `_silence_and_vinyl()` — 3-phase: dither ratio inside silent passages (±50), digital-void-above-cutoff (+20), vinyl surface noise (random autocorr + stable energy, −40) and click counting (−10)
- `_psychoacoustic_artifacts()` — pre-echo (HF energy before transients), HF filterbank aliasing correlation, MP3 32-band subband comb (spectral autocorrelation at 689 Hz multiples)
- `_cassette_source()` — Rule 11 veto: tape hiss + natural slope + wow/flutter; score ≥30 subtracts 40 and disarms the segment vote
- `_spectral_sparsity()` — psychoacoustically zeroed bins (<-95 dB rel) *below* the cutoff
- `_ultrasonic_envelope_correlation()` — Pearson corr of mid-band vs high-band envelopes; exposes anti-forensic fake HF noise injection when combined with a collapsed auCDtect bound
- `_fft_band_extract()` — zero-phase FFT brickwall band isolation. **Use this, not Butterworth, for noise-floor measurement**: IIR skirts (~24 dB/oct) leak loud music into a quiet band ~0.1 octave away
- `analyse()` — orchestrates the 11-rule flow, combines into **Main Score (0–100)**

**Main Score verdict thresholds** (`_verdict()`): ≥86 LIKELY_LOSSY · ≥55 SUSPICIOUS · ≥31 CAUTION · ≥11 LIKELY_GENUINE · <11 GENUINE. Natively lossy formats (.mp3 etc.) keep the informational CAUTION path. scipy missing → base engine only + warning caveat.

**Verdict labels**: GENUINE, LIKELY_GENUINE, CAUTION, SUSPICIOUS, LIKELY_LOSSY, INCONCLUSIVE

**Measured detection performance** (synthetic pink-noise fixtures, encode→FLAC/ALAC, 44.1k and 48k): MP3 64–320 kbps → 88–100 LIKELY_LOSSY; AAC 96–192 → 88–100; Opus 64–192 → 88 (CELT fingerprint); Vorbis q2–q4 → 91–100; 24/96→MP3 320→ALAC case study → 88; half-spliced → 45 CAUTION with regions; 16/44.1 upsampled to 24/48 → 45 SUSPICIOUS (mirror mode) and to 24/96 → 75 SUSPICIOUS (notch mode), both labelled "Sample-rate counterfeit"; genuine/vinyl-sim/dark-master/cassette-sim/mono all 0. **Known misses**: AAC ≥256 kbps and Vorbis q6+ keep full bandwidth on pink noise (no wall to detect) — real-music artifacts are the only path there.

#### 3. **Report Generation** (`build_report()` / `build_info_report()`)
- `build_report()` → Full forensic analysis. Subprocess-bound extractors (loudness graph, SoX stats, spectrogram, bit-depth probe) run in a ThreadPoolExecutor while the SpectralEngine crunches on the main thread; byproduct metrics reuse the engine's decode. Wall time ≈ max(engine, loudness) instead of the sum of everything (~6× faster).
- `build_info_report()` → Lightweight metadata only (no spectral analysis)
- Batch mode processes files concurrently (`--workers`, default auto up to 3); reports print in order with per-file timing in the footer.

#### 4. **Display & Formatting**
Terminal output with ANSI color codes:
- `C` class: Color palette (C.RED, C.GREEN, C.YELLOW, C.BLUE, C.ORANGE, C.CYAN, C.GOLD, C.GREY, C.WHITE)
- `_c(colour, text)` — Wrap text in color codes
- `_rule()` / `_section()` / `_subsection()` — Header formatting
- `_kv(key, value)` — Key-value pair alignment
- Color functions like `_peak_colour()`, `_lufs_colour()`, `_crest_colour()` — Metric-specific coloring logic
- `print_report()` → Main formatted terminal output (handles all sections)
- `print_batch_summary()` → Album-level rollup when multiple files analyzed

#### 5. **CLI Entry Point**
`main()` function handles:
- Argument parsing (files, --json, --fast, --info)
- Tool availability checks (ffmpeg, sox, mediainfo must be in PATH)
- Batch file processing
- JSON serialization (via `_report_to_dict()`)

## Metric Reference & Thresholds

### Loudness Metrics
- **LUFS (Integrated)**: Target -14 (Spotify/Tidal) to -16 (Apple Music)
- **DR Score**: DR5-8 typical for modern mastered audio; DR10+ = highly dynamic
- **Crest Factor**: 3-8 dB normal; <3 dB = heavily compressed
- **True Peak**: -1 dBTP typical; >0 dBTP = clipping

### Spectral Metrics (Forensics)
- **HF Cutoff**: Normal is 20 kHz (Nyquist at 48kHz) or gradual rolloff starting ~19 kHz. Suspicious if <18 kHz unless explained by format.
- **Cutoff Variance**: Low variance (<1k Hz²) = rigid/encoded; High variance (>100k Hz²) = natural/organic
- **Cliff Sharpness**: Gradual <2 dB/bin; sharp cliff >15 dB/bin suggests hard filter
- **HF Energy Ratio**: <0.005 with low cutoff = suspicious; >0.015 = healthy
- **Banding Score**: >0.95 with low HF energy = quantization artifacts (lossy indicator)
- **Side Anomaly**: <0.15 = healthy stereo; >0.7 = severe anomaly (joint stereo artifacts)
- **Entropy**: Low <0.3 = tonal music (normal); high >0.6 with low cutoff = lossy noise-shaping

### Advanced DSP Metrics (scipy suite)
- **Cliff Depth**: dB drop across ±400 Hz around cutoff. >35 dB = codec wall; natural fades are single digits
- **Segment Vote**: walled clips / 7. Majority = +55 (the single strongest lossy signal)
- **auCDtect Bound**: avg ≥85% Nyquist = lossless-like; <16.5 kHz = statistical void (+25)
- **HF Phase Entropy**: <4.0 structured; >4.5 bits with depressed cutoff = quantized phase (+10); max is log2(36)≈5.17
- **Spectral Sparsity**: <0.05 dense; >0.30 below cutoff = codec bin-zeroing (+10)
- **Ultrasonic Corr.**: >0.6 HF breathes with music; <0.15 + collapsed bound = injected fake noise (+15)
- **Silence Dither Ratio**: >0.3 codec hash in silence (+50). **Asymmetric**: clean silence (<0.15) is only worth −30 and ONLY with full bandwidth + no wall evidence — lossy encoders zero out digital silence too, so clean silence must never cancel wall evidence (this exact bug let a 24/96→MP3 320→ALAC chain score 18)
- **Void above cutoff**: band rms < −85 dBFS (FFT-extracted, cutoff+800 → Nyquist−100) = digital upscale (+20) and arms the adaptive segment-vote wall
- **Vinyl**: random (autocorr <0.3), stable (var <5 dB) noise above cutoff = analog (−40); 5–50 clicks/min confirms (−10)
- **Resample Check**: foreign-Nyquist fingerprint (wall / notch / aliased mirror at exactly 22,050 / 24,000 / 44,100 / 48,000 Hz) = upsample (+45, verdict forced to ≥SUSPICIOUS). Mirror corr threshold 0.35 (measured: swr fake 0.49, genuine 0.003). Bit-depth trailing-zero analysis is structurally blind to resampled upscales — this rule is the detector for them
- **Cassette score**: ≥30/80 **and R11A hiss actually found** = tape source veto (−40, disarms segment vote). A cassette without tape hiss doesn't exist; slope/flutter alone must not veto.
- **Bit depth (Source Integrity)**: effective bits from trailing zeros — ✓ all claimed bits used · ⚠ effective ≤ claimed−8 = padded upscale · ~ in-between = bit-shifted gain / fixed-point chain. **Display gate**: a green "✓ Verified" is demoted to an orange "~ unverifiable" whenever the engine already called the file fake — `resample_detected OR fake_hires OR verdict ∈ {SUSPICIOUS, LIKELY_LOSSY}` — because any transcode/upsample regenerates the low-order bits (float decode + requantization, or interpolation), so trailing-zero analysis is structurally blind there. Keyed on its own broad condition (in `print_report`, not the bit-depth function) so a reassuring green never appears on a fake even if the bandwidth rule's own thresholds are conservative. **Fake-hires (96k container, 20k bandwidth ceiling) is the case where the resample detector itself misses — see `_is_fake_hires_bandwidth`**
- **Fake Hi-Res Bandwidth**: ≥88.2k container whose content ends in a >25 dB cliff at <60% of Nyquist with a <−80 dB void above = upsampled, even when no foreign-Nyquist resampler tell survived (codec lowpass erased it). +20 and verdict forced to ≥SUSPICIOUS. Shown in the Resample Check row as "⚠ fake hi-res — …"

### Live progress & ETA
`_Status` renders a single thread-safe stderr line (TTY only): `⏳ [done/total] file: stage ▰▰▰▱▱▱ ~Ns`. `_STAGE_PROGRESS` maps each stage name to its cumulative progress fraction (profiled on the 4-min fake path — keep in sync if stages are added/reordered); ETA = `elapsed·(1−p)/p`, self-calibrating to the machine (no absolute speed model). Batch mode adds `batch ~Ns left` = max(active ETAs) + queued·avg_file_time/workers (`begin(total, workers)`). The engine reports stages via the `status` callback param of `analyse()`; `build_report` reports probe/finalize stages.

### Detection case study (regression-test this scenario)
24/96 FLAC → 320 kbps MP3 → 24-bit ALAC originally scored 18/GENUINE: segment vote correctly failed 7/7 (+55) but the clean-silence credit (−50) cancelled it and its early-return skipped the void check (+20). Fix: clean silence is asymmetric evidence (gated −30), no early return. Now scores 78/SUSPICIOUS. Synthetic fixture: pink noise 24/96 with a 3 s muted span → lame 320 → alac.

## Common Modifications

### Adding a new metric
1. Add extraction function (calls external tool, returns parsed value)
2. Add field to appropriate dataclass (AudioTechnical, LoudnessProfile, SpectralAnalysis)
3. Add coloring logic (e.g., `_new_metric_colour()`)
4. Add to `print_report()` output section
5. Update metric reference in README if appropriate

### Adjusting scoring thresholds
Spectral verdict logic lives in `SpectralEngine.analyse()`. Adjust:
- Lossy/natural indicator score weights (the `SCORE_*` constants) for the base engine
- Main Score deltas in the "Advanced 11-rule forensic suite" block inside `analyse()`
- Individual metric interpretation strings (the `_interp_*` static methods)
- Verdict thresholds in `_verdict()` (86/55/31/11 on the 0–100 main score)
After any threshold change, re-run `python test_dsp.py`, regenerate fixtures with `python testdata/make_fixtures.py` (gitignored; pink-noise sources → MP3/AAC/Opus/Vorbis encodes at 44.1k+48k → FLAC fakes + genuine/vinyl/cassette/dark/mono/spliced controls + `fake_upsampled_2448/2496` sample-rate counterfeits; `--only combos` adds lossy→upscale **cascades** `fake_cascade_*` (mp3320→48/96, aac128/256→96), a pure bit-depth pad `fake_bitpad_2444` (16→24, no resample), `fake_upsampled_4896` (48→96), and the genuine `genuine_96k24` hi-res control), and run `python testdata/measure.py` to confirm the separation holds: every walled fake ≥86, upsampled/cascade fakes ≥45 SUSPICIOUS, every genuine control (incl. `genuine_96k24`) ≤20. `testdata/profile.py` prints per-stage engine timings.

### Changing output format
Modify `print_report()` and `print_batch_summary()` for terminal output, or adjust `_report_to_dict()` for JSON schema changes.

### Supporting new audio formats
Add format to either:
- `_SOX_UNSUPPORTED` set if SoX doesn't support it natively (will auto-convert to WAV via ffmpeg)
- Or add tool-specific handling to extractors if special parsing is needed

## File Structure
```
audio-forensic/
├── audio_forensic.py      (~1,900 lines — all code, no external modules)
├── test_dsp.py            (synthetic-signal verification of every DSP metric)
├── requirements.txt       (numpy + scipy dependencies + tool notes)
├── README.md             (user-facing documentation)
├── LICENSE               (MIT)
├── CLAUDE.md             (this file)
└── testdata/             (gitignored: make_fixtures.py, measure.py, profile.py + generated fixtures)
```

## Testing & Debugging Tips

- Use `--info` flag to test metadata extraction without full analysis
- Use `--fast` to speed up development (only processes first 60 seconds)
- Use `--json` to inspect raw data structure (easier than parsing colored terminal output)
- Temporarily add `print()` statements in extractors to debug tool output parsing
- External tools (ffmpeg, sox) output to stderr; check `_run()` calls if a metric fails silently
- SpectralEngine logs findings in `SpectralAnalysis.evidence` / `natural_evidence` lists for debugging verdict logic

## Important Notes

- **Numpy optional**: Code gracefully degrades if numpy unavailable; SpectralEngine won't run but other analysis continues
- **Scipy optional**: Without scipy the advanced suite is skipped (warning to stderr, caveat in the report); the base spectral engine still scores into the main scale
- **Performance**: ~3.4 s wall for a genuine 4-min FLAC, ~4.6 s for a fake (engine 2.3 s / 3.5 s). One decode + one cached STFT feeds everything; scipy's pocketfft runs with `workers=-1`; `_fft_band_extract` runs float32 and caches the forward rfft per signal (keyed on length + content samples — void/cassette/vinyl all band-slice the same cap); envelopes are rectified+smoothed |x|·π/2 (NOT Hilbert — the analytic transform cost a full complex-FFT round trip for identical transient localization); the side-channel STFT runs at 4× hop with the mid frames strided to match; 9B aliasing uses direct dot-product Pearson on non-overlapping 5 s segments (corrcoef copied arrays); auCDtect moments run float32; time-domain analyses capped at `TIME_DOMAIN_CAP_S` (180 s). Reference timings live in the footer of every report ("Analysed in X.Xs")
- **Windows pipes**: `main()` reconfigures stdout/stderr to UTF-8 — cp1252 pipes crashed on the report's box-drawing glyphs
- **Tool dependencies**: All external tool calls return gracefully on failure; missing tools are caught upfront in `main()`
- **No temp files**: all SoX input arrives via stdin pipes from ffmpeg
- **Platform support**: Uses `where` (Windows) vs. `which` (Unix) for tool detection; Bash/PowerShell compatible
- **Batch processing**: No file size limits; files run concurrently (`--workers`), output stays in argument order
- **Spectrogram generation**: SoX rendering (better visual, user preference), ffmpeg showspectrumpic fallback; keep `-y` at 2ⁿ+1
