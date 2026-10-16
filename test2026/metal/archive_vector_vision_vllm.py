"""Archive one terminal full matched cohort and the entire pinned source closure."""
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile
root=Path(sys.argv[1]).resolve();folder=Path(sys.argv[2]).resolve();target=Path(sys.argv[3]).resolve();assert folder.is_relative_to(root/'logs') and not target.exists()
pid=json.loads((folder/'pid.json').read_text())['pid'];state=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat='],capture_output=True,text=True);assert state.returncode in (0,1) and (not state.stdout.strip() or state.stdout.strip().startswith('Z'));status=json.loads((folder/'exit.json').read_text());assert isinstance(status['exit_code'],int)

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

selection_path=root/'work/vector-vision-next-comparator-selection.json';selection=json.loads(selection_path.read_text());files={'evidence/selection.json':selection_path};omitted={}
for name,sha in selection['source_manifest'].items():
 p=root/name;assert digest(p)==sha,name;files['evidence/source/'+name]=p
source=root/selection['native_source_root'];scripts=source/'test2026/metal'
# Preserve the original audit's basename lookup and the exact external driver.
for p in scripts.glob('*'):
 if p.is_file() and p.suffix in ('.py','.mm','.metal','.h'):files['evidence/source/test2026/metal/'+p.name]=p
for p in Path(__file__).parent.glob('*.py'):files['evidence/source/test2026/metal/'+p.name]=p
for p in folder.rglob('*'):
 if not p.is_file():continue
 name=p.relative_to(root).as_posix()
 if p.suffix in ('.json','.py','.log','.txt'):files['evidence/'+name]=p
 else:omitted[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
for name,info in selection['asset_manifest'].items():
 p=root/name;assert p.stat().st_size==info['bytes'] and digest(p)==info['sha256'];omitted[name]=info
for p in (root/'models').glob('*/config.json'):files['evidence/'+p.relative_to(root).as_posix()]=p
for build in selection['builds'].values():
 p=root/build['directory']/'libgarnet.dylib';assert digest(p)==build['sha256'];omitted[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=digest(p))
proof=dict(files={},omitted=omitted,jobs={folder.name:status},scope='One complete matched FP32 cohort; pinned d52 source/library, original input/caps/stops, actual comparator GPU dtypes, sequential full outputs and independent clocks. Full two-cohort objective remains unproven.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
 for name,p in sorted(files.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 for name,value in (('evidence/files.json',proof['files']),('evidence/manifest.json',proof)):
  raw=(json.dumps(value,indent=2)+'\n').encode();item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(path=str(target),sha256=digest(target),files=len(files),status=status)))
