"""Preserve the numeric hardware capability probe and pinned sources."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
folder=root/'logs/mixed-accumulator-capability-20261015'
assert json.loads((folder/'exit.json').read_text())['exit_code']==0
entries={};omitted={}
for p in folder.rglob('*'):
    if not p.is_file():continue
    name=str(p.relative_to(root))
    if p.suffix in ('.json','.log','.txt','.py','.mm','.metal'):entries[name]=p
    else:omitted[name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for name in ('mixed_accumulator_capability_probe.mm','run_mixed_accumulator_capability.py','queue_mixed_accumulator_capability.py','run_metal_counter_capabilities.py','run_decode_subsimd_probe.py'):
    p=root/'source/test2026/metal'/name;entries[str(p.relative_to(root))]=p
proof=dict(files={},omitted=omitted,scope='Hardware precision capability only; no model or speed evidence')
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(entries.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(entries),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
