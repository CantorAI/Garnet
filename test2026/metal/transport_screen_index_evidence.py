"""Small transport of a frozen full archive, borrowing only identical captured arrays."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
archive=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();assert not target.exists()

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

included={};borrowed={}
with tarfile.open(archive,'r:gz') as source,tarfile.open(target,'w:gz') as destination:
 for item in source:
  assert item.isfile();raw=source.extractfile(item).read();info=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  if item.name.startswith('evidence/captured-arrays/'):
   borrowed[item.name]=info;continue
  included[item.name]=info;record=tarfile.TarInfo(item.name);record.size=len(raw);destination.addfile(record,io.BytesIO(raw))
 record=dict(original_full_archive_sha256=digest(archive),original_full_archive_bytes=archive.stat().st_size,included=included,borrowed_captured_arrays=borrowed,scope='Transport only. Restore every borrowed array from a separately hash-verified previous cohort before a complete full-manifest audit. This does not change the frozen original archive or mathematical/performance records.',full_goal_met=False)
 raw=(json.dumps(record,indent=2)+'\n').encode();item=tarfile.TarInfo('transport-manifest.json');item.size=len(raw);destination.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(path=str(target),sha256=digest(target),bytes=target.stat().st_size,included_files=len(included),borrowed_files=len(borrowed),original_full_archive_sha256=record['original_full_archive_sha256'])))
