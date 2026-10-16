"""Archive only terminal reset component jobs, including failed source snapshots."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]);files={}
folders=sorted((r/'logs').glob('state-zero-gpu-probe*'));assert folders
for folder in folders:
    assert (folder/'exit.json').is_file(),'Live job: '+str(folder)
    for path in folder.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.txt','.log','.py','.mm','.tsv'):
            files['evidence/logs/'+str(path.relative_to(r/'logs'))]=path
for name in ('state_zero_gpu.mm','run_state_zero_gpu_probe.py','archive_state_zero_gpu.py','background_comparison.py'):
    files['evidence/source/'+name]=r/'source/test2026/metal'/name
manifest={}
with tarfile.open(destination,'w:gz') as archive:
    for name,path in sorted(files.items()):
        raw=path.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/files.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(manifest),archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))
