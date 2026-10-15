"""Preserve terminal joint-reset component runs and raw executed profiles."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();entries={};jobs={};omitted={}
for folder in sorted((root/'logs').glob('joint-state-zero-*20261015')):
    jobs[folder.name]=json.loads((folder/'exit.json').read_text())
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.log','.txt','.tsv','.py','.mm'):
            entries[str(p.relative_to(root))]=p
    binary=folder/'jobs/joint_state_zero_gpu'
    if binary.exists():omitted[str(binary.relative_to(root))]=dict(bytes=binary.stat().st_size,sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    meta=json.loads((folder/'jobs/metadata.json').read_text())
    for model,c in meta['configs'].items():
        p=root/'models'/model/'config.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==c['sha256'];entries[str(p.relative_to(root))]=p
assert jobs
proof=dict(files={},jobs=jobs,omitted=omitted,scope='Reset/read component only; no inference or comparator evidence')
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(entries.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(entries),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),jobs=jobs)))
