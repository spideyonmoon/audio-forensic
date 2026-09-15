# Accuracy changes — September 2026

## AAC false-positive fixes

The AAC MDCT quantization-error detector previously treated a silent stereo side
channel as a perfect integer lattice. Clean dual-mono noise scored 1.00 and a
pure tone scored approximately 0.74 in the diagnostic controls.

The detector now selects active anchors independently per channel, requires at
least four anchors, and excludes weak or unpopulated bands from lattice votes.
The same dual-mono control scores approximately 0.022 and the tone scores 0.00.
Files with insufficient active spectral frames return INCONCLUSIVE.

## Vorbis transform-grid detector

`SpectralEngine._vorbis_grid` searches decoded L/R channels using the Vorbis
sine-of-sine-squared window and common 2048/256-sample block configuration.
Reconstructing L/R matters because a mid-channel average can fill quantized zeros.

Up to 12 probe positions within the first 180 seconds are searched at every
sample phase. Near-zero coefficients between 1 and 19 kHz are compared with
off-grid transforms. After short-block sequences, long-block alignment can shift
by multiples of the 128-sample short hop. Equivalent phases are pooled before
voting, including in the background estimate. Requiring one fixed 1024-sample
long-block phase missed both real music and independently generated transients.

At least four probes and half the active probes must support the pattern. Each
supporting probe needs at least 0.03 excess zero occupancy and occupancy at least
three times its background plus 0.01. A qualifying aggregate establishes a score
floor of 55 (SUSPICIOUS), rather than adding a second penalty for spectral zeros.
The result exposes score, supporting/tested probe counts, channel and explanation
in JSON and the terminal report. Probe support is not a duration-coverage estimate.

Window and overlap reference: [Xiph Vorbis I specification](https://www.xiph.org/vorbis/doc/Vorbis_I_spec.html).
This is an original heuristic based on codec structure, not a claimed reproduction
of a published forensic classifier. The newly supplied 2015 Lossless Audio Checker
paper was not implemented during this work.

## Validation

```text
python -X utf8 test_dsp.py
python -X utf8 -m unittest -v test_accuracy test_vorbis
```

- 79 existing DSP checks and 13 accuracy regression test methods.
- AAC 256/320 kbps at 44.1/48 kHz remains detectable in the supported TNS-disabled fixtures.
- Stereo libvorbis q4/q6/q8/q10 converted to 24-bit FLAC at 44.1/48 kHz is detected.
- Tested q10 fixtures remain detectable after trimming 137 samples and changing gain.
- Independent transient-heavy Vorbis q6/q8 fixtures reproduce the former fixed-phase
  miss; corrected detection succeeds while their unencoded source stays unflagged.
- Genuine controls include noise, tones, harmonics, hard/gentle lowpasses,
  clipping, 16-bit rounding, silence and dual mono.

Two user-supplied tracks were also analyzed (their asserted Vorbis ancestry was
not independently verified against original encodes):

| Track | Main score | Vorbis excess | Supporting probes |
| --- | ---: | ---: | ---: |
| CHANYEOL, Punch — Stay With Me - Instrumental | 55 | 0.149 | 7/11, right channel |
| Masha — Ja Re Jare Ure Jare Pakhi | 55 | 0.141 | 6/11, right channel |

## Interpretation and limits

The score is heuristic evidence, not a probability. A zero-score explanation now
states that source history is unverified; the existing GENUINE label remains for
compatibility. Negative Vorbis results explicitly do not exclude Vorbis ancestry.

The Vorbis search currently covers 44.1/48 kHz and the common block configuration.
Transition windows, other block sizes, resampling, noise/dither, and subsequent
processing can erase evidence. Its extra transform search adds runtime. Controlled
fixtures and two positive examples do not establish real-world precision/recall;
that requires independently verified masters, matched encodes and held-out music.
