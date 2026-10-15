"""Archive complete diagnostic outputs, raw counter files and exact source copy."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();entries={};omitted=[]
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def add(p):
    name=str(p.relative_to(root));entries[name]=(p,dict(path=name,bytes=p.stat().st_size,sha256=digest(p)))
folder=root/'logs/counter-trace-file-profiles-20261015'
assert json.loads((folder/'exit.json').read_text())['exit_code']==0
for p in folder.rglob('*'):
    if not p.is_file():continue
    if p.suffix in ('.json','.jsonl','.log','.txt','.py'):add(p)
    else:omitted.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
identities=json.loads((folder/'jobs/source-identities.json').read_text())
for name,sha in identities['diagnostic_source_sha256'].items():
    p=root/'counter-trace-file-source'/name;assert digest(p)==sha;add(p)
original=root/'source/src/backends/metal/metal_context.mm';assert digest(original)==identities['original_metal_context_sha256'];add(original)
meta=json.loads((folder/'jobs/metadata.json').read_text())
for name,sha in meta['source_sha256'].items():
    p=root/'source/test2026/metal'/name;assert digest(p)==sha;add(p)
for name in ('run_joined_prefill_checks.py','compare_vllm_metal.py','launch_decode_matrix_probe.py','launch_counter_trace_profiles.py','archive_counter_trace_profiles.py'):add(root/'source/test2026/metal'/name)
assets={}
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    reference=root/'logs/combined-prefill-reset-confirm-20261015/jobs'/(model+'-b1-attempt0')/'metadata.json';add(reference)
    expected=json.loads(reference.read_text())['assets'][model]
    assets[model]={name:digest(root/'models'/model/name) for name in ('model.safetensors','config.json','tokenizer.json')}
    assert assets[model]==expected;add(root/'models'/model/'config.json')
assert digest(root/'build-counter-trace-file/libgarnet.dylib')==meta['library_sha256']
proof=target.with_suffix('.manifest.json');proof.write_bytes((json.dumps(dict(files=[e for _,e in entries.values()],omitted=omitted,library_sha256=meta['library_sha256'],assets=assets,scope='Compute-pass diagnostic only; sampled clocks are not matched serving performance'),indent=2)+'\n').encode())
with tarfile.open(target,'w:gz') as archive:
    archive.add(proof,arcname='manifest.json')
    for p,e in entries.values():archive.add(p,arcname=e['path'])
print(json.dumps(dict(files=len(entries),bytes=target.stat().st_size,sha256=digest(target))))
