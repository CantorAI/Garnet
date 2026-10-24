"""External, small native wait dump; never restart or alter the study worker."""
import argparse
import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
import msvcrt
from pathlib import Path, PureWindowsPath
import zipfile

p = argparse.ArgumentParser(); p.add_argument("--pid", type=int, required=True); args = p.parse_args()
root = Path("D:/CantorAI/5090-study-20261023")
out = root / "results/full-attention-policy-control-wait-attempt1"
archive = out.with_name(out.name + "-evidence.zip")
assert not out.exists() and not archive.exists()
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
dbghelp = ctypes.WinDLL("dbghelp", use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
kernel.GetProcessTimes.restype = wintypes.BOOL
dbghelp.MiniDumpWriteDump.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.HANDLE, wintypes.DWORD,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
dbghelp.MiniDumpWriteDump.restype = wintypes.BOOL
handle = kernel.OpenProcess(0x0400 | 0x0010, False, args.pid)
if not handle: raise ctypes.WinError(ctypes.get_last_error())
try:
    name = ctypes.create_unicode_buffer(32768); size = wintypes.DWORD(len(name))
    if not kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    assert PureWindowsPath(name.value) == PureWindowsPath(root / "build/bin/xlang3.exe")
    created, exited, system, user = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
        raise ctypes.WinError(ctypes.get_last_error())
    birth = ((created.dwHighDateTime << 32) | created.dwLowDateTime) / 1e7 - 11644473600
    # Match the observed original process instance, not merely a recycled PID.
    assert abs(birth - 1792864449.051) < 0.05, birth
    library = root / "build/bin/xlang3.exe"
    library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    assert library_hash == "2e2db12a2dbc1134a3a4c662c080b91d500b37af488661630d4d580e7684dc10"
    out.mkdir()
    dump = out / "worker.dmp"
    # Normal stacks + thread information + memory-region metadata + unloaded
    # module list. Deliberately excludes MiniDumpWithFullMemory (0x2).
    flags = 0x1000 | 0x0800 | 0x0020
    with dump.open("xb") as f:
        file_handle = wintypes.HANDLE(msvcrt.get_osfhandle(f.fileno()))
        success = bool(dbghelp.MiniDumpWriteDump(handle, args.pid, file_handle, flags, None, None, None))
        error = ctypes.get_last_error() if not success else 0
    receipt = {"status": "wait_dump_captured" if success else "dump_api_failed", "pid": args.pid,
        "process_birth_unix": birth, "process_image": name.value, "dump_flags": flags,
        "error": error, "executable_sha256": library_hash, "dump_bytes": dump.stat().st_size,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "External diagnostic of a live untimed study worker; normal stack dump, no full address-space/weight dump, termination, restart, source or DLL mutation. Does not prove a cause or measure model speed."}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
    raw = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    raw["capture_native_policy_wait_dump.py"] = Path(__file__).read_bytes()
    raw["xlang3.exe"] = library.read_bytes()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in raw.items(): z.writestr(n, b)
        z.writestr("manifest.json", json.dumps({"files": {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()}}, indent=2))
    print(json.dumps({"receipt": receipt, "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2))
    if not success: raise ctypes.WinError(error)
finally:
    kernel.CloseHandle(handle)
