"""Create one study-owned SSD scratch slot, preserving source/models on D:."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path("D:/CantorAI/5090-study-20261023")
out = root / "results/study-ssd-engine-cache"
assert not out.exists()
cache = (Path(os.environ["LOCALAPPDATA"]) / "CantorAI/5090-engine-cache").resolve()
assert cache.drive.upper() == "C:"
assert shutil.disk_usage("C:/").free >= 64 * 1024**3
command = "$p=@(Get-Partition -DriveLetter C,D|Select-Object DriveLetter,DiskNumber,Size);$d=@(Get-PhysicalDisk|Select-Object DeviceId,FriendlyName,MediaType,BusType,Size);[pscustomobject]@{partitions=$p;disks=$d}|ConvertTo-Json -Depth 5"
raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", command])
storage = json.loads(raw.decode("utf-8-sig"))
partition = next(v for v in storage["partitions"] if v["DriveLetter"] == "C")
disk = next(v for v in storage["disks"] if str(v["DeviceId"]) == str(partition["DiskNumber"]))
assert disk["MediaType"] == "SSD" and disk["BusType"] == "NVMe"
owner = cache / "study-owner.json"
if cache.exists(): assert not any(cache.iterdir()), "Refuse to adopt an existing nonempty folder"
cache.mkdir(parents=True, exist_ok=True)
record = {"status": "ssd_engine_scratch_prepared", "study": str(root), "cache_root": str(cache),
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "storage": storage,
    "cleanup_policy": "Only private garnet-cuda-graph-* engine files created here; closed study workers only; preserve source/models/raw results. Keep the same owner root for subsequent tests.",
    "scope": "Temporary engine placement only. Does not change checkpoint/model math or prove a warm speedup or the cause of every observed I/O wait."}
owner.write_text(json.dumps(record, indent=2), encoding="utf8")
out.mkdir()
(out / "receipt.json").write_text(json.dumps(record, indent=2), encoding="utf8")
print(json.dumps(record, indent=2))
