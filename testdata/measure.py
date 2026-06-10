#!/usr/bin/env python3
"""Run the SpectralEngine on every fixture and tabulate the forensic metrics.

Usage: python measure.py [glob ...]   (default: fake_*.flac fake_*.m4a genuine_*.flac)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import audio_forensic as af  # noqa: E402


def probe(path: Path) -> tuple[int, int, float]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=sample_rate,channels,duration", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    return int(s["sample_rate"]), int(s.get("channels", 2)), float(s.get("duration", 0) or 0)


def main() -> None:
    patterns = sys.argv[1:] or ["fake_*.flac", "fake_*.m4a", "genuine_*.flac"]
    files = sorted({p for pat in patterns for p in HERE.glob(pat)})
    hdr = f"{'file':34} {'sr':>6} {'cutoff':>7} {'cliff':>6} {'nf_abv':>7} {'bound':>7} {'seg':>9} {'sprs':>5} {'sil':>5} {'main':>4}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for p in files:
        sr, ch, dur = probe(p)
        eng = af.SpectralEngine(p, sr, channels=ch, claimed_duration=dur)
        t0 = time.perf_counter()
        r = eng.analyse()
        dt = time.perf_counter() - t0
        seg = f"{r.segment_walled}/{r.segment_total}@{r.segment_wall_hz/1000:.1f}" if r.segment_walled >= 0 else "n/a"
        print(f"{p.name:34} {sr:>6} {r.cutoff_hz:>7,.0f} {r.cliff_depth_db:>6.1f} {r.nf_above_cutoff_db:>7.1f} "
              f"{r.auc_avg_bound_freq:>7,.0f} {seg:>9} {r.spectral_sparsity:>5.2f} {r.silence_ratio:>5.2f} "
              f"{r.main_score:>4}  {r.verdict_label} ({dt:.1f}s)")


if __name__ == "__main__":
    main()
