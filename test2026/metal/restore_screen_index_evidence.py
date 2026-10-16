"""Verify transport and restore identical captured arrays; audit the complete original manifest."""
from pathlib import Path
import hashlib,json,shutil,sys,tarfile
archive=Path(sys.argv[1]).resolve();expected=sys.argv[2];donor=Path(sys.argv[3]).resolve();out=Path(sys.argv[4]).resolve();assert not out.exists()

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

assert digest(archive)==expected;out.mkdir(parents=True)
with tarfile.open(archive) as tar:
 names=set()
 for item in tar:
  assert item.isfile() and item.name not in names;names.add(item.name);p=(out/item.name).resolve();assert p.is_relative_to(out);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(item).read())
transport=json.loads((out/'transport-manifest.json').read_text());proof=json.loads((out/'evidence/manifest.json').read_text());copied=0
for name,info in transport['included'].items():assert (out/name).stat().st_size==info['bytes'] and digest(out/name)==info['sha256']
for name,info in transport['borrowed_captured_arrays'].items():
 assert name.startswith('evidence/captured-arrays/');source=(donor/name).resolve();target=(out/name).resolve();assert source.is_relative_to(donor/'evidence/captured-arrays') and target.is_relative_to(out/'evidence/captured-arrays') and not target.exists()
 assert source.stat().st_size==info['bytes'] and digest(source)==info['sha256'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);copied+=1
for name,info in proof['files'].items():
 p=(out/name).resolve();assert p.is_relative_to(out) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
result=dict(transport_sha256_verified=expected,original_full_archive_sha256_as_reported_by_mac=transport['original_full_archive_sha256'],original_full_archive_bytes_verified_on_windows=False,verified_original_manifest_files=len(proof['files']),borrowed_identical_array_files=copied,donor=str(donor),scope='All original manifest members independently byte/hash verified after captured-array deduplication. Original full gzip bytes remain on Mac and were not transferred or verified as a gzip blob on Windows.',full_goal_met=False)
(out/'transport-verification.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
