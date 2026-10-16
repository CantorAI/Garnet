"""Archive terminal success or failure before changing any learned component job source."""
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
for name in ('archive_prepared_union_mlp_learned.py','verify_prepared_union_mlp_learned.py','analyze_prepared_union_mlp_learned.py'):
 p=Path(__file__).with_name(name);files['evidence/tools/'+name]=p
proof=dict(files={},omitted=omitted,jobs={folder.name:status},hardware={k:subprocess.check_output(['sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize','hw.model')},versions=dict(macos=subprocess.check_output(['sw_vers'],text=True),compiler=subprocess.check_output(['xcrun','clang++','--version'],text=True)),scope='Private learned component, all numeric records and predetermined endpoints; no native inference claim.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
 for name,p in sorted(files.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=status)))
