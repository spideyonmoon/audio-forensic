#!/usr/bin/env python3
"""Peak memory of the whole process tree — the figure a container limit enforces.

`measure_memory.py` reports the engine process alone. On a memory-capped host (Render's
free tier, a small VPS, Docker --memory) the limit applies to every process in the
container, so the ffmpeg/sox children the engine spawns count too. This walks the process
tree with the Toolhelp32 API and sums resident memory for the engine plus all descendants.

Usage
-----
    python testdata/container_peak.py audio_forensic.py FILE [NAME=VALUE ...]

    # how much does the extractor fan-out cost the container?
    python testdata/container_peak.py audio_forensic.py track.flac AF_EXTRACTORS=4
    python testdata/container_peak.py audio_forensic.py track.flac AF_EXTRACTORS=1

Windows-only, like measure_memory.py (Toolhelp32 + GetProcessMemoryInfo).
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import threading
import time

TH32CS_SNAPPROCESS = 0x00000002
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
MB = 1024 * 1024


class PROCESSENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)), ("th32ModuleID", wt.DWORD),
                ("cntThreads", wt.DWORD), ("th32ParentProcessID", wt.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
                ("szExeFile", ctypes.c_char * 260)]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    """Exactly the Windows layout: dropping a field silently shifts everything after it."""
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


kernel32 = ctypes.WinDLL("kernel32")
psapi = ctypes.WinDLL("psapi")
kernel32.CreateToolhelp32Snapshot.restype = wt.HANDLE
kernel32.CreateToolhelp32Snapshot.argtypes = (wt.DWORD, wt.DWORD)
kernel32.Process32First.argtypes = (wt.HANDLE, ctypes.POINTER(PROCESSENTRY32))
kernel32.Process32Next.argtypes = (wt.HANDLE, ctypes.POINTER(PROCESSENTRY32))
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.OpenProcess.argtypes = (wt.DWORD, wt.BOOL, wt.DWORD)
kernel32.CloseHandle.argtypes = (wt.HANDLE,)
psapi.GetProcessMemoryInfo.argtypes = (wt.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wt.DWORD)
psapi.GetProcessMemoryInfo.restype = wt.BOOL


def _process_table() -> dict[int, int]:
    """{pid: parent_pid} for every process on the machine."""
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == INVALID_HANDLE_VALUE:
        return {}
    table: dict[int, int] = {}
    entry = PROCESSENTRY32()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
    try:
        ok = kernel32.Process32First(snap, ctypes.byref(entry))
        while ok:
            table[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            ok = kernel32.Process32Next(snap, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snap)
    return table


def _live_working_set(pid: int) -> int:
    """Current resident size — summed live across the tree, since each process's own
    *peak* would double-count memory that was never held at the same time."""
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return 0
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return 0
        return counters.WorkingSetSize
    finally:
        kernel32.CloseHandle(handle)


def _tree_memory(root_pid: int) -> tuple[int, int]:
    """(sum of live working sets across the tree, engine's own live working set)."""
    table = _process_table()
    children: dict[int, list[int]] = {}
    for pid, ppid in table.items():
        children.setdefault(ppid, []).append(pid)
    stack, seen, total, own = [root_pid], set(), 0, 0
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        size = _live_working_set(pid)
        total += size
        if pid == root_pid:
            own = size
        stack.extend(children.get(pid, ()))
    return total, own


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    script, target = sys.argv[1], sys.argv[2]
    env = dict(os.environ)
    for assignment in sys.argv[3:]:
        k, _, v = assignment.partition("=")
        env[k] = v

    proc = subprocess.Popen([sys.executable, "-X", "utf8", script, "--json", target],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    peak_tree = peak_own = 0
    stop = threading.Event()

    def sampler():
        nonlocal peak_tree, peak_own
        while not stop.is_set():
            tree, own = _tree_memory(proc.pid)
            peak_tree, peak_own = max(peak_tree, tree), max(peak_own, own)
            stop.wait(0.1)

    t0 = time.perf_counter()
    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    proc.wait()
    stop.set()
    thread.join()

    label = " ".join(sys.argv[3:]) or "(engine defaults)"
    print(f"  {label:<24} tree peak={peak_tree / MB:>5.0f} MB   "
          f"engine only={peak_own / MB:>5.0f} MB   wall={time.perf_counter() - t0:>5.1f}s")
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    sys.exit(main())
