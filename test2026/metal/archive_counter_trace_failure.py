"""Preserve successful capability/build and the diagnostic log-channel failure."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();entries={}
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def add(p):
    name=str(p.relative_to(root));entries[name]=(p,dict(path=name,bytes=p.stat().st_size,sha256=digest(p)))
cap=root/'logs/metal-counter-capabilities-20261015';failed=root/'logs/counter-trace-profiles-20261015'
assert json.loads((cap/'exit.json').read_text())['exit_code']==0
assert json.loads((failed/'exit.json').read_text())['exit_code']==1
assert 'JSONDecodeError' in (failed/'output.log').read_text()
for folder in (cap,failed):
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.txt','.log','.py','.mm'):add(p)
for p in (root/'counter-trace-source').rglob('*'):
    if p.is_file():add(p)
for name in ('metal_counter_capabilities_probe.mm','run_metal_counter_capabilities.py','launch_metal_counter_capabilities.py','counter_trace_helpers.inc','make_counter_trace_source.py','run_counter_trace_profiles.py','launch_counter_trace_profiles.py','run_joined_prefill_checks.py','compare_vllm_metal.py','launch_decode_matrix_probe.py','archive_counter_trace_failure.py'):add(root/'source/test2026/metal'/name)
proof=target.with_suffix('.manifest.json');proof.write_bytes((json.dumps(dict(files=[e for _,e in entries.values()],candidate_sha256=digest(root/'build-counter-trace/libgarnet.dylib'),scope='Failed diagnostic JSON parsing after successful hardware capability, build, unit and native workers; no new serving comparison'),indent=2)+'\n').encode())
with tarfile.open(target,'w:gz') as archive:
    archive.add(proof,arcname='manifest.json')
    for p,e in entries.values():archive.add(p,arcname=e['path'])
print(json.dumps(dict(files=len(entries),sha256=digest(target),bytes=target.stat().st_size)))
