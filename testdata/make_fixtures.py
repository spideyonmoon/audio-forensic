#!/usr/bin/env python3
"""Fixture generator: pink-noise sources -> lossy encodes -> FLAC/ALAC fakes,
plus genuine-family controls. All files land in testdata/. Gitignored, throwaway.

Usage: python make_fixtures.py [--only mp3|aac|opus|vorbis|controls]
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DUR = 60.0


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise via spectral shaping of white noise."""
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    X /= np.sqrt(f)
    x = np.fft.irfft(X, n=n)
    return (x / np.max(np.abs(x)) * 0.5).astype(np.float32)


def stereo_pink(sr: int, seconds: float = DUR, seed: int = 7, mute_span: tuple | None = None) -> np.ndarray:
    n = int(sr * seconds)
    rng = np.random.default_rng(seed)
    common = pink_noise(n, rng)
    l = 0.8 * common + 0.2 * pink_noise(n, rng)
    r = 0.8 * common + 0.2 * pink_noise(n, rng)
    x = np.stack([l, r], axis=1)
    if mute_span:
        s, e = (int(t * sr) for t in mute_span)
        x[s:e] = 0.0
    return x.astype(np.float32)


def write_pcm(x: np.ndarray, sr: int, out: Path, codec: str = "pcm_s16le", out_sr: int | None = None) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "pipe:0",
           "-c:a", codec]
    if out_sr: cmd += ["-ar", str(out_sr)]
    cmd.append(str(out))
    subprocess.run(cmd, input=x.tobytes(), check=True)


def run(cmd: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error"] + cmd, check=True)


def transcode(src: Path, enc_args: list[str], lossy: Path, fake: Path, fake_args: list[str] | None = None) -> None:
    run(["-i", str(src)] + enc_args + [str(lossy)])
    run(["-i", str(lossy)] + (fake_args or ["-c:a", "flac"]) + [str(fake)])


def main() -> None:
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    HERE.mkdir(exist_ok=True)

    src44 = HERE / "src_44k16.wav"
    src48 = HERE / "src_48k24.wav"
    src96 = HERE / "src_96k24_mute.wav"
    if not src44.exists(): write_pcm(stereo_pink(44100), 44100, src44)
    if not src48.exists(): write_pcm(stereo_pink(48000, seed=8), 48000, src48, codec="pcm_s24le")
    if not src96.exists(): write_pcm(stereo_pink(96000, seed=9, mute_span=(25, 28)), 96000, src96, codec="pcm_s24le")

    if only in (None, "mp3"):
        for br in (64, 96, 128, 160, 192, 256, 320):
            transcode(src44, ["-c:a", "libmp3lame", "-b:a", f"{br}k"],
                      HERE / f"_tmp_mp3_{br}.mp3", HERE / f"fake_mp3_{br}.flac")

    if only in (None, "aac"):
        for br in (96, 128, 192, 256):
            transcode(src44, ["-c:a", "aac", "-b:a", f"{br}k"],
                      HERE / f"_tmp_aac_{br}.m4a", HERE / f"fake_aac_{br}.flac")

    if only in (None, "opus"):
        for br in (64, 96, 128, 160, 192):
            # Opus is 48 kHz internally. Two fake targets: native 48k and resampled-back 44.1k.
            lossy = HERE / f"_tmp_opus_{br}.opus"
            run(["-i", str(src44), "-c:a", "libopus", "-b:a", f"{br}k", str(lossy)])
            run(["-i", str(lossy), "-c:a", "flac", str(HERE / f"fake_opus_{br}_48k.flac")])
            run(["-i", str(lossy), "-ar", "44100", "-c:a", "flac", str(HERE / f"fake_opus_{br}_44k.flac")])

    if only in (None, "vorbis"):
        for q in (2, 4, 6, 8):
            transcode(src44, ["-c:a", "libvorbis", "-q:a", str(q)],
                      HERE / f"_tmp_vorbis_q{q}.ogg", HERE / f"fake_vorbis_q{q}.flac")

    if only in (None, "controls"):
        # Genuine controls
        run(["-i", str(src44), "-c:a", "flac", str(HERE / "genuine_44k16.flac")])
        run(["-i", str(src48), "-c:a", "flac", str(HERE / "genuine_48k24.flac")])
        run(["-i", str(src44), "-ac", "1", "-c:a", "flac", str(HERE / "genuine_mono.flac")])
        # Dark master: gentle 1-pole lowpass at 16 kHz (6 dB/oct — no codec wall)
        run(["-i", str(src44), "-af", "lowpass=f=16000:p=1", "-c:a", "flac", str(HERE / "genuine_dark_master.flac")])
        # Vinyl sim: 2-pole LPF at 18k + stable white hiss + clicks (numpy)
        sr = 44100
        x = stereo_pink(sr, DUR, seed=11)
        rng = np.random.default_rng(12)
        hiss = rng.standard_normal(x.shape).astype(np.float32) * 10 ** (-58 / 20)
        clicks = np.zeros_like(x)
        for t in rng.uniform(1, DUR - 1, size=20):  # 20 clicks/min
            i = int(t * sr)
            clicks[i : i + 30] = rng.standard_normal((30, 2)).astype(np.float32) * 0.05
        vin = HERE / "_tmp_vinyl_raw.wav"
        write_pcm((x + hiss + clicks).astype(np.float32), sr, vin, codec="pcm_f32le")
        run(["-i", str(vin), "-af", "lowpass=f=18000:p=2", "-c:a", "flac", str(HERE / "genuine_vinyl_sim.flac")])
        # Cassette sim: gentle 14k rolloff + tape hiss
        cas_hiss = rng.standard_normal(x.shape).astype(np.float32) * 10 ** (-48 / 20)
        cas = HERE / "_tmp_cassette_raw.wav"
        write_pcm((x + cas_hiss).astype(np.float32), sr, cas, codec="pcm_f32le")
        run(["-i", str(cas), "-af", "lowpass=f=14000:p=1", "-c:a", "flac", str(HERE / "genuine_cassette_sim.flac")])
        # Case study: 24/96 with mute span -> MP3 320 -> 24-bit ALAC
        mp3 = HERE / "_tmp_case_320.mp3"
        run(["-i", str(src96), "-c:a", "libmp3lame", "-b:a", "320k", str(mp3)])
        run(["-i", str(mp3), "-c:a", "alac", "-sample_fmt", "s32p", str(HERE / "fake_casestudy_alac.m4a")])
        # Spliced: first half genuine, second half MP3-128 decode
        dec = subprocess.run(["ffmpeg", "-v", "error", "-i", str(HERE / "_tmp_mp3_128.mp3"),
                              "-f", "f32le", "-ac", "2", "-ar", "44100", "pipe:1"],
                             capture_output=True, check=True)
        lossy_half = np.frombuffer(dec.stdout, dtype=np.float32).reshape(-1, 2)
        gen = stereo_pink(44100, DUR, seed=7)
        n_half = int(30 * 44100)
        spliced = np.concatenate([gen[:n_half], lossy_half[n_half : int(DUR * 44100)]])
        sp = HERE / "_tmp_spliced_raw.wav"
        write_pcm(np.ascontiguousarray(spliced, dtype=np.float32), 44100, sp, codec="pcm_f32le")
        run(["-i", str(sp), "-c:a", "flac", "-sample_fmt", "s16", str(HERE / "fake_spliced_half.flac")])

    print("fixtures done:", len(list(HERE.glob("*.flac"))) + len(list(HERE.glob("*.m4a"))), "test files")


if __name__ == "__main__":
    main()
