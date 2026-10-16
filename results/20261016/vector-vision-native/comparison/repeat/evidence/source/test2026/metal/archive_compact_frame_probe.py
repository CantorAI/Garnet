"""Freeze terminal CPU preparation evidence without copying raw image buffers."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();folder=root/'logs/compact-frame-repair-probe-20261014'
status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==0
files={}
failed=root/'logs/compact-frame-probe-20261014';assert json.loads((failed/'exit.json').read_text())['exit_code']==1
for p in failed.rglob('*'):
 if p.is_file() and p.suffix in ('.py','.json','.log','.txt'):files['evidence/logs/'+str(p.relative_to(root/'logs'))]=p
for p in folder.rglob('*'):
 if p.is_file() and p.suffix in ('.py','.json','.log','.txt'):files['evidence/logs/'+str(p.relative_to(root/'logs'))]=p
for name in ('frame_preprocessing_compact.py','test_compact_frame_preprocessing.py','run_compact_frame_probe.py','queue_compact_frame_probe.py','archive_compact_frame_probe.py'):
 files['evidence/source/'+name]=root/'source/test2026/metal'/name
manifest={}
with tarfile.open(target,'w:gz') as tar:
 for name,p in files.items():
  raw=p.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 raw=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/files.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
