#!/usr/bin/env python3
"""Per-stage timing profile of SpectralEngine.analyse() and build_report().

Usage: python profile.py file1 [file2 ...]
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
    for name in sys.argv[1:]:
        p = HERE / name if (HERE / name).exists() else Path(name)
        sr, ch, dur = probe(p)
        eng = af.SpectralEngine(p, sr, channels=ch, claimed_duration=dur)
        marks: list[tuple[str, float]] = []
        t0 = time.perf_counter()
        r = eng.analyse(status=lambda s: marks.append((s, time.perf_counter())))
        t_end = time.perf_counter()
        print(f"\n{p.name} ({dur:.0f}s audio) — engine total {t_end - t0:.2f}s, score {r.main_score}/{r.verdict_label}")
        for i, (stage, ts) in enumerate(marks):
            nxt = marks[i + 1][1] if i + 1 < len(marks) else t_end
            print(f"  {stage:28} {nxt - ts:6.2f}s")
        # Full report wall time (parallel extractors + engine)
        t1 = time.perf_counter()
        af.build_report(p)
        print(f"  {'build_report wall':28} {time.perf_counter() - t1:6.2f}s")


if __name__ == "__main__":
    main()
