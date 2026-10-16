"""Safely extract frozen research evidence and verify every manifest member."""
from pathlib import Path
import hashlib,json,sys,tarfile

archive=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
expected=sys.argv[3]
with archive.open('rb') as stream:
    actual=hashlib.file_digest(stream,'sha256').hexdigest()
assert actual==expected,(actual,expected)
if '--verify-only' not in sys.argv[4:]:
    out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive) as tar:
        seen=set()
        for member in tar:
            assert member.isfile() and member.name not in seen
            seen.add(member.name)
            path=(out/member.name).resolve();assert path.is_relative_to(out)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(tar.extractfile(member).read())
proof=json.loads((out/'evidence/manifest.json').read_text())
for name,info in proof['files'].items():
    # The small controller archive uses root-relative manifest names;
    # larger validation archives include the evidence/ prefix explicitly.
    path=(out/name if name.startswith('evidence/') else out/'evidence'/name).resolve()
    assert path.is_relative_to(out)
    raw=path.read_bytes()
    assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256'],name
result=dict(archive_sha256=actual,verified_files=len(proof['files']),
            scope='Archive bytes and all manifest files only; does not prove full validation or performance',full_goal_met=False)
(out/'archive-verification.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result))
