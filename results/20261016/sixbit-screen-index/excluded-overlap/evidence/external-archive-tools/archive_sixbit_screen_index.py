"""Freeze a terminal six-bit-index component, source bytes and captured input/full-logit arrays."""
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile
root=Path(sys.argv[1]).resolve();folder=Path(sys.argv[2]).resolve();target=Path(sys.argv[3]).resolve();assert folder.is_relative_to(root/'logs') and not target.exists()
pid=int(json.loads((folder/'pid.json').read_text())['pid']);state=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat=,comm='],capture_output=True,text=True);assert state.returncode in (0,1) and (not state.stdout.strip() or state.stdout.strip().startswith('Z'));status=json.loads((folder/'exit.json').read_text());assert isinstance(status['exit_code'],int)

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

files={};omitted={}
for p in folder.rglob('*'):
 if not p.is_file():continue
 name=p.relative_to(root).as_posix()
 if p.suffix in ('.json','.py','.h','.hpp','.cpp','.mm','.metal','.make','.log','.txt'):files['evidence/'+name]=p
 else:omitted[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
meta_path=folder/'jobs/metadata.json'
if meta_path.exists():
 meta=json.loads(meta_path.read_text())
 for name,sha in meta['source_sha256'].items():assert digest(Path(name))==sha,name
 for model,asset in meta['assets'].items():
  source=Path(asset['directory']);info=asset['manifest'];p=source/'raw.bin';assert digest(p)==info['raw_sha256'];omitted[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=digest(p))
  for query in info['queries']:
   for suffix,key in (('.input.bin','input_sha256'),('.reference.bin','reference_sha256')):
    p=source/(query['name']+suffix);assert digest(p)==query[key];files['evidence/captured-arrays/'+model+'/'+p.name]=p
  p=root/'models'/model/'config.json';files['evidence/models/'+model+'/config.json']=p
  p=root/'models'/model/'model.safetensors';assert digest(p)==info['checkpoint_sha256'];omitted[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=digest(p))
for name in ('archive_sixbit_screen_index.py','verify_sixbit_screen_index.py'):files['evidence/external-archive-tools/'+name]=Path(__file__).with_name(name)
proof=dict(files={},omitted=omitted,jobs={folder.name:status},scope='Isolated captured-query rejection index, original full/retained-bit/unsafe/safety assertions and own-null clocks only. No current full-model throughput/TTFT or fresh vLLM result.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
 for name,p in sorted(files.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(path=str(target),sha256=digest(target),files=len(files),status=status)))
