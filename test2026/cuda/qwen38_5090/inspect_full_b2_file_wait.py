"""Read file-handle/storage metadata for the original live process; no deletion."""
import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil

root = Path("D:/CantorAI/5090-study-20261023")
out = root / "results/native-canonical-fp8-file-wait-attempt1"
assert not out.exists(); out.mkdir()
def record(event, **fields):
    value = {"event": event, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}
    with (out / "observations.jsonl").open("a", encoding="utf8") as f: f.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; k.OpenProcess.restype = wintypes.HANDLE
k.CloseHandle.argtypes = [wintypes.HANDLE]; k.CloseHandle.restype = wintypes.BOOL
k.GetCurrentProcess.restype = wintypes.HANDLE
k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]; k.QueryFullProcessImageNameW.restype = wintypes.BOOL
k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4; k.GetProcessTimes.restype = wintypes.BOOL
k.DuplicateHandle.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; k.DuplicateHandle.restype = wintypes.BOOL
k.GetFileType.argtypes = [wintypes.HANDLE]; k.GetFileType.restype = wintypes.DWORD
k.GetFinalPathNameByHandleW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]; k.GetFinalPathNameByHandleW.restype = wintypes.DWORD
k.GetFileSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)]; k.GetFileSizeEx.restype = wintypes.BOOL
pid = 14080
process = k.OpenProcess(0x0400 | 0x0040, False, pid)
if not process: raise ctypes.WinError(ctypes.get_last_error())
copied = wintypes.HANDLE()
try:
    image = ctypes.create_unicode_buffer(32768); count = wintypes.DWORD(len(image))
    if not k.QueryFullProcessImageNameW(process, 0, image, ctypes.byref(count)): raise ctypes.WinError(ctypes.get_last_error())
    assert PureWindowsPath(image.value) == PureWindowsPath(root / "build/bin/xlang3.exe")
    times = [wintypes.FILETIME() for _ in range(4)]
    if not k.GetProcessTimes(process, *[ctypes.byref(t) for t in times]): raise ctypes.WinError(ctypes.get_last_error())
    birth = ((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime) / 1e7 - 11644473600
    assert abs(birth - 1792857388.642) < 0.05
    record("process_instance_verified", pid=pid, birth_unix=birth, image=image.value)
    # 0x580 is the handle in the externally captured NtWriteFile context.
    if not k.DuplicateHandle(process, wintypes.HANDLE(0x580), k.GetCurrentProcess(), ctypes.byref(copied), 0, False, 2): raise ctypes.WinError(ctypes.get_last_error())
    kind = k.GetFileType(copied); record("handle_type_observed", source_handle="0x580", file_type=kind)
    final = None; length = None
    if kind == 1:
        buffer = ctypes.create_unicode_buffer(32768)
        size = k.GetFinalPathNameByHandleW(copied, buffer, len(buffer), 0)
        if not size or size >= len(buffer): raise ctypes.WinError(ctypes.get_last_error())
        final = buffer.value; record("file_path_observed", path=final)
        number = ctypes.c_longlong()
        if k.GetFileSizeEx(copied, ctypes.byref(number)): length = number.value
        record("file_size_observed", bytes=length)
    disks = {}
    for drive in ("C:/", "D:/"):
        try:
            u = shutil.disk_usage(drive); disks[drive] = {"total": u.total, "used": u.used, "free": u.free}
        except OSError as error: disks[drive] = {"error": str(error)}
    record("disk_usage_observed", disks=disks)
    caches = []
    for path in (root / "tmp").iterdir():
        if not path.name.startswith("garnet-cuda-graph-"): continue
        st = path.lstat()
        if getattr(st, "st_file_attributes", 0) & 0x400: continue
        if not path.is_dir(): continue
        files = []
        for f in path.iterdir():
            fs = f.lstat()
            files.append({"name": f.name, "bytes": fs.st_size, "attributes": getattr(fs, "st_file_attributes", 0)})
        caches.append({"path": str(path), "ctime": st.st_ctime, "mtime": st.st_mtime, "files": files})
    result = {"status": "file_wait_metadata_observed", "pid": pid, "birth_unix": birth,
        "handle": "0x580", "file_type": kind, "path": final, "file_bytes": length,
        "disks": disks, "study_tmp_caches": caches, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Read-only process/file/disk/cache metadata. No file deletion, source/runtime mutation, worker termination, restart or model timing."}
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    record("completed", caches=len(caches), observed_cache_bytes=sum(f["bytes"] for c in caches for f in c["files"]))
finally:
    if copied: k.CloseHandle(copied)
    k.CloseHandle(process)
