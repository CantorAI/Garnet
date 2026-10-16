"""Freeze a terminal isolated GPU value-tile experiment, raw logs, sources and asset identities."""
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile
root=Path(sys.argv[1]).resolve();folder=Path(sys.argv[2]).resolve();target=Path(sys.argv[3]).resolve();assert folder.is_relative_to(root/'logs') and not target.exists()
pid=int(json.loads((folder/'pid.json').read_text())['pid']);process=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat=,comm='],capture_output=True,text=True);assert process.returncode in (0,1) and (not process.stdout.strip() or process.stdout.strip().startswith('Z'))
status=json.loads((folder/'jobs/exit.json').read_text());assert isinstance(status['exit_code'],int);files={};omitted={}

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

for p in folder.rglob('*'):
 if not p.is_file():continue
 name=p.relative_to(root).as_posix()
 if p.suffix in ('.py','.mm','.metal','.make','.json','.log','.txt'):files['evidence/'+name]=p
 else:omitted[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
config=root/'models/Qwen3.5-0.8B/config.json';files['evidence/config.json']=config
for name in ('archive_delta_value_tiles_gpu.py','verify_delta_value_tiles_gpu.py','analyze_delta_value_tiles_gpu.py'):
 files['evidence/external-archive-tools/'+name]=Path(__file__).with_name(name)
proof=dict(files={},omitted=omitted,jobs={folder.name:status},terminal_process=process.stdout.strip(),scope='Isolated original-kernel bit checks and component clocks only. No native integration, model inference speed or vLLM target claim.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
 for name,p in sorted(files.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(path=str(target),sha256=digest(target),files=len(files),omitted_files=len(omitted),status=status)))
