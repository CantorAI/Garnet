"""Preserve terminal CPU strip-resize trials and executed helper sources."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();entries={};jobs={}
for folder in sorted((root/'logs').glob('striped-frame-probe-*20261015')):
    jobs[folder.name]=json.loads((folder/'exit.json').read_text())
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix in ('.py','.json','.log','.txt'):entries['evidence/'+str(p.relative_to(root))]=p
for name in ('frame_preprocessing_striped.py','test_striped_frame_preprocessing.py','run_striped_frame_probe.py','launch_striped_frame_probe.py','archive_striped_frame_probe.py','verify_striped_frame_probe.py'):
    entries['evidence/source/'+name]=root/'source/test2026/metal'/name
assert jobs
proof=dict(jobs=jobs,files={})
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(entries.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(entries),jobs=jobs,sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
