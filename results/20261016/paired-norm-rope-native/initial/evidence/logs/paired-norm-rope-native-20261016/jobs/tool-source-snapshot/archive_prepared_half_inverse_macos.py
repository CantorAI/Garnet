"""Freeze actual Apple CPU inverse stage; compiled payloads omitted by full identity."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
folder=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();assert not target.exists();status=json.loads((folder/'exit.json').read_text());assert isinstance(status['exit_code'],int)
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
files={};omitted={}
for p in folder.rglob('*'):
 if not p.is_file():continue
 name=p.relative_to(folder).as_posix()
 if p.suffix in ('.json','.log','.py','.cpp','.h'):files['evidence/'+name]=p
 else:omitted[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
meta=json.loads((folder/'metadata.json').read_text());compiler=Path(meta['compiler']);assert digest(compiler)==meta['compiler_sha256'];omitted['compiler/clang++']=dict(bytes=compiler.stat().st_size,sha256=meta['compiler_sha256'])
for name in ('archive_prepared_half_inverse_macos.py','verify_prepared_half_inverse_macos.py'):
 p=Path(__file__).with_name(name);files['evidence/tools/'+name]=p
proof=dict(files={},omitted=omitted,jobs={'macos-exhaustive-cpu':status},scope='Actual Apple-clang CLZ mathematical prerequisite only; no GPU/native/model speed claim.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
 for name,p in sorted(files.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=status)))
