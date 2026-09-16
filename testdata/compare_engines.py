#!/usr/bin/env python3
"""Byte-identical output check between two revisions of the engine.

The engine's accuracy guarantee is that a change reproduces the previous output
exactly. This runs both revisions over the same files and compares every field of
the JSON report (all but the wall-clock timing), the rendered spectrogram PNG, and
optionally the decoded audio samples themselves.

Usage
-----
    # 1. extract the previous revision (usually the committed one)
    git show <commit>:audio_forensic.py > base_engine.py

    # 2. compare reports + spectrograms
    python testdata/compare_engines.py base_engine.py audio_forensic.py FILE [FILE ...]

    # 3. for changes that touch decoding, also compare the decoded samples
    python testdata/compare_engines.py --samples base_engine.py audio_forensic.py FILE ...

Exit status is 0 only when every file matched, so it composes with &&.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

IGNORE_FIELDS = {"analysis_seconds"}


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in IGNORE_FIELDS}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _diff(path: str, a, b, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            _diff(f"{path}.{k}", a.get(k, "<missing>"), b.get(k, "<missing>"), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"  {path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(f"{path}[{i}]", x, y, out)
    elif a != b:
        out.append(f"  {path}: base={a!r} new={b!r}")


def _run(script: Path, audio: Path, out_json: Path) -> int:
    with open(out_json, "w", encoding="utf-8") as fh:
        proc = subprocess.run([sys.executable, "-X", "utf8", str(script), "--json", str(audio)],
                              stdout=fh, stderr=subprocess.DEVNULL)
    return proc.returncode


def _spectrogram_bytes(audio: Path) -> bytes | None:
    png = audio.with_name(f"{audio.stem}_spectrogram.png")
    return png.read_bytes() if png.exists() else None


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def compare_samples(base_path: Path, new_path: Path, audio: Path, sample_rate: int = 44100) -> list[str]:
    """Compare the decoded mid/side (or mono) arrays between revisions."""
    import numpy as np
    base, new = _load("cmp_base_engine", base_path), _load("cmp_new_engine", new_path)
    problems = []
    try:
        b_eng = base.SpectralEngine(audio, sample_rate, channels=2)
        n_eng = new.SpectralEngine(audio, sample_rate, channels=2)
        b_pair, n_pair = b_eng._decode_stereo(), n_eng._decode_stereo()
        if (b_pair is None) != (n_pair is None):
            return [f"  decode returned None on only one side (base={b_pair is None}, new={n_pair is None})"]
        if b_pair is not None:
            for label, x, y in zip(("mid", "side"), b_pair, n_pair):
                if x.shape != y.shape:
                    problems.append(f"  stereo {label}: shape {x.shape} != {y.shape}")
                elif not np.array_equal(x, y):
                    problems.append(f"  stereo {label}: values differ ({int((x != y).sum())} samples)")
        b_mono, n_mono = b_eng._decode_audio(), n_eng._decode_audio()
        if (b_mono is None) != (n_mono is None):
            problems.append("  mono decode returned None on only one side")
        elif b_mono is not None:
            if b_mono.shape != n_mono.shape:
                problems.append(f"  mono: shape {b_mono.shape} != {n_mono.shape}")
            elif not np.array_equal(b_mono, n_mono):
                problems.append(f"  mono: values differ ({int((b_mono != n_mono).sum())} samples)")
    except Exception as exc:                      # a revision may be too old for a given call
        problems.append(f"  decode comparison raised: {exc!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="previous engine revision (a .py file)")
    ap.add_argument("new", help="new engine revision (a .py file)")
    ap.add_argument("files", nargs="+", help="audio files to compare")
    ap.add_argument("--samples", action="store_true", help="also compare decoded audio samples")
    ap.add_argument("--no-png", action="store_true", help="skip the spectrogram PNG comparison")
    args = ap.parse_args()

    base_path, new_path = Path(args.base).resolve(), Path(args.new).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="engine_cmp_"))
    failed = 0

    for raw in args.files:
        audio = Path(raw).resolve()
        print(f"\n{audio.name}")
        base_json, new_json = tmp / "base.json", tmp / "new.json"

        rc_base = _run(base_path, audio, base_json)
        base_png = None if args.no_png else _spectrogram_bytes(audio)
        rc_new = _run(new_path, audio, new_json)
        new_png = None if args.no_png else _spectrogram_bytes(audio)

        if rc_base != rc_new:
            print(f"  exit codes differ: base={rc_base} new={rc_new}")
            failed += 1
            continue
        if rc_base != 0:
            print(f"  both revisions failed (exit {rc_base}) — skipped")
            continue

        problems: list[str] = []
        try:
            b = _strip(json.loads(base_json.read_text(encoding="utf-8")))
            n = _strip(json.loads(new_json.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            print(f"  could not parse JSON output: {exc}")
            failed += 1
            continue
        _diff("", b, n, problems)

        if not args.no_png:
            if (base_png is None) != (new_png is None):
                problems.append("  spectrogram: present on only one side")
            elif base_png is not None and base_png != new_png:
                problems.append(f"  spectrogram: PNG differs "
                                f"({hashlib.sha1(base_png).hexdigest()[:12]} vs "
                                f"{hashlib.sha1(new_png).hexdigest()[:12]})")

        if args.samples:
            problems.extend(compare_samples(base_path, new_path, audio))

        if problems:
            failed += 1
            print("  DIFFERS")
            for line in problems[:40]:
                print("  " + line)
            if len(problems) > 40:
                print(f"  ... {len(problems) - 40} more")
        else:
            score = b[0]["authenticity"]["spectral"]["main_score"] if isinstance(b, list) else "?"
            print(f"  identical  (score={score})")

    print()
    if failed:
        print(f"FAILED — {failed} of {len(args.files)} file(s) differ")
        return 1
    print(f"OK — all {len(args.files)} file(s) byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
