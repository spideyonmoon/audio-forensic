"""Memory measurement harness for audio_forensic.py.

Mirrors the methodology of MEMORY_REPORT.md: runs the CLI in a fresh Python
process, samples PeakWorkingSetSize and PrivateUsage (commit charge) every
250 ms via GetProcessMemoryInfo, and reports the peaks.

Usage:
    python testdata/measure_memory.py [--json] [--batch] -- FILE [FILE ...]
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import subprocess
import sys
import threading
import time


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


psapi = ctypes.WinDLL("psapi")
kernel32 = ctypes.WinDLL("kernel32")
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.OpenProcess.argtypes = (wt.DWORD, wt.BOOL, wt.DWORD)
kernel32.CloseHandle.argtypes = (wt.HANDLE,)
psapi.GetProcessMemoryInfo.argtypes = (wt.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wt.DWORD)
psapi.GetProcessMemoryInfo.restype = wt.BOOL


def sample(pid: int) -> tuple[int, int] | None:
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        pmc = PROCESS_MEMORY_COUNTERS_EX()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        if not psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return None
        # Working set ≈ resident RAM; PrivateUsage = commit charge, matching
        # MEMORY_REPORT.md's "private memory".
        return pmc.PeakWorkingSetSize, pmc.PrivateUsage
    finally:
        kernel32.CloseHandle(h)


MB = 1024 * 1024


def measure(cmd: list[str], label: str) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak_ws, peak_private = 0, 0
    stop = threading.Event()

    def sampler():
        nonlocal peak_ws, peak_private
        while not stop.is_set():
            s = sample(proc.pid)
            if s:
                peak_ws = max(peak_ws, s[0])
                peak_private = max(peak_private, s[1])
            stop.wait(0.25)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    proc.wait()
    stop.set()
    th.join()
    wall = time.perf_counter() - t0
    return {
        "label": label,
        "exit": proc.returncode,
        "wall_s": round(wall, 1),
        "peak_ws_mb": round(peak_ws / MB),
        "peak_private_mb": round(peak_private / MB),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--batch", action="store_true", help="one process for all files (batch mode)")
    ap.add_argument("--script", default="", help="analyse with this script instead of ./audio_forensic.py")
    args = ap.parse_args()

    script = args.script or str(__import__("pathlib").Path(__file__).parent.parent / "audio_forensic.py")
    rows = []
    if args.batch:
        cmd = [sys.executable, script, "--json", "--workers", "3", *args.files]
        rows.append(measure(cmd, "BATCH all files"))
    else:
        for f in args.files:
            cmd = [sys.executable, script, "--json", f]
            rows.append(measure(cmd, f))

    for r in rows:
        print(f"{r['label']:<60.60} exit={r['exit']} ws={r['peak_ws_mb']:>5} MB "
              f"private={r['peak_private_mb']:>5} MB wall={r['wall_s']:>5}s")
    if args.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
