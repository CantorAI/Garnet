"""Freeze terminal combined validation and serving without mixing old timings."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
folder=root/'logs/combined-prefill-reset-20261015';entries={}
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def add(p):
    assert p.is_file() and p.resolve().is_relative_to(root)
    name=str(p.relative_to(root));entries[name]=(p,dict(path=name,bytes=p.stat().st_size,sha256=digest(p)))
jobs={name:json.loads((folder/name).read_text()) for name in ('exit.json','checks-exit.json','native-exit.json')}
assert all(v['exit_code']==0 for v in jobs.values())
for p in folder.rglob('*'):
    if p.is_file() and p.suffix in ('.json','.log','.txt','.py','.f32','.executed'):add(p)
for area in (root/'prefill-stream-staging',root/'logs/prefill-stream-original-source-20261014'):
    for p in area.rglob('*'):
        if p.is_file():add(p)
manifest=json.loads((root/'prefill-stream-staging/manifest.json').read_text())
checks=json.loads((folder/'checks/metadata.json').read_text())
identities=json.loads((folder/'source-identities.json').read_text())
assert identities['manifest']==manifest
for name,sha in {**manifest,**checks['source_sha256']}.items():
    p=root/'source'/name;assert digest(p)==sha;add(p)
for name,sha in identities['helpers'].items():
    p=root/'source/test2026/metal'/name;assert digest(p)==sha;add(p)
# Retain the established transitive helper set, but archive current bytes and
# only this experiment's logs. None of its predecessor timings is admitted.
prior=json.loads((root/'prefill-stream-repaired-confirm-native-evidence.tar.manifest.json').read_text())
for entry in prior['files']:
    if entry['path'].startswith('source/test2026/metal/') and entry['path'].endswith('.py'):
        add(root/entry['path'])
for name in ('archive_combined_prefill_reset.py','test_prefix_reset_session_reference.py','run_prefix_reset_session_reference.py','dirty_prefix_reset_test.cpp'):
    add(root/'source/test2026/metal'/name)
add(root/'source/src/backends/metal/compiled_graph.h')
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):add(root/'models'/model/'config.json')
candidate=root/'build-combined-prefill-reset/libgarnet.dylib';control=root/'build-global-half-alpha/libgarnet.dylib'
assert digest(candidate)==checks['candidate_sha256']
assert digest(control)=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
proof=target.with_suffix('.manifest.json')
proof.write_bytes((json.dumps(dict(files=[e for _,e in entries.values()],jobs=jobs,candidate_sha256=digest(candidate),control_sha256=digest(control)),indent=2)+'\n').encode())
with tarfile.open(target,'w:gz') as archive:
    archive.add(proof,arcname='manifest.json')
    for p,e in entries.values():archive.add(p,arcname=e['path'])
print(json.dumps(dict(files=len(entries),bytes=target.stat().st_size,sha256=digest(target))))
