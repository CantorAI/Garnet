"""Install only the explicitly staged candidate into the isolated Mac source."""
import hashlib,json,tarfile,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();source=r/'source';manifest=json.loads((r/'native-bfloat-body-source-manifest.json').read_text())
original={'src/backends/metal/metal_context.mm':'3dc37728607efd57c1d70a5d28a3a00dfce9787e6072f7b4e33ceb7e484d74f9','cmake/GarnetMetal.cmake':'ce346c0068a3ad7f0c267085904249c409025fd4e3e3ada29052479767813b8e'}
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert all(digest(source/n)==v for n,v in original.items()),'Isolated source changed before candidate installation'
backup=r/'logs/native-bfloat-body-original-source-20261013';backup.mkdir(exist_ok=False)
for name in original:
    p=backup/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((source/name).read_bytes())
(backup/'manifest.json').write_text(json.dumps(original,indent=2))
allowed=set(original)|{'test2026/metal/run_native_bfloat_body_checks.py','test2026/metal/compare_native_bfloat_body.py','test2026/metal/run_native_bfloat_body_profiles.py'}
assert set(manifest)==allowed
with tarfile.open(r/'native-bfloat-body-source.tar.gz') as archive:
    members=archive.getmembers();assert {m.name for m in members}==allowed and all(m.isfile() for m in members)
    for member in members:
        raw=archive.extractfile(member).read();assert hashlib.sha256(raw).hexdigest()==manifest[member.name]
        (source/member.name).write_bytes(raw)
assert all(digest(source/n)==v for n,v in manifest.items())
print('Preserved original hashes; installed five verified isolated candidate files')
