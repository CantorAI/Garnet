"""Freeze any terminal Paired norm-RoPE private-context stage before repairs; never omit failure logs."""
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile
root=Path(sys.argv[1]).resolve();folder=Path(sys.argv[2]).resolve();target=Path(sys.argv[3]).resolve();assert folder.is_relative_to(root/'logs') and not target.exists()
status=json.loads((folder/'exit.json').read_text());assert isinstance(status['exit_code'],int)
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
files={};omitted={}
for p in folder.rglob('*'):
 if not p.is_file():continue
 name=p.relative_to(root).as_posix()
 if p.suffix in ('.json','.log','.py','.h','.mm','.cpp','.txt','.make'):files['evidence/'+name]=p
 else:omitted[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
for pattern in ('archive_paired_norm_rope.py','verify_paired_norm_rope*.py','analyze_paired_norm_rope*.py','confirm_paired_norm_rope*.py'):
 for p in Path(__file__).parent.glob(pattern):files['evidence/tools/'+p.name]=p
proof=dict(files={},omitted=omitted,jobs={folder.name:status},hardware={k:subprocess.check_output(['sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize','hw.model')},versions=dict(macos=subprocess.check_output(['sw_vers'],text=True),compiler=subprocess.check_output(['xcrun','clang++','--version'],text=True)),scope='Private Paired norm-RoPE component only; no complete-model/native speed admission.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
 for name,p in sorted(files.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=status)))
