"""Archive terminal full-gate, session-recovery and native ABBA evidence."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();entries={};omitted={};jobs={}
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for name in ('fused-head-argmax-checks-20261015','fused-head-argmax-native-20261015'):
    folder=root/'logs'/name;jobs[name]=json.loads((folder/'exit.json').read_text())
    for p in folder.rglob('*'):
        if not p.is_file():continue
        if p.suffix in ('.json','.log','.txt','.py','.mm','.cpp','.h','.metal','.executed'):
            entries[str(p.relative_to(root))]=p
        else:omitted[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=digest(p))
staging=root/'fused-head-argmax-staging'
original=json.loads((staging/'original.json').read_text());candidate=json.loads((staging/'manifest.json').read_text())
for name,sha in candidate.items():
    p=root/'source'/name;assert digest(p)==sha;entries[str(p.relative_to(root))]=p
for p in (root/'logs/fused-head-argmax-original-source-20261015').rglob('*'):
    if p.is_file():entries[str(p.relative_to(root))]=p
helpers=('archive_fused_head_argmax_inference.py','run_guarded_fused_head_argmax_checks.py','launch_fused_head_argmax_checks.py','run_guarded_global_half_alpha_checks.py','run_global_half_alpha_checks.py','run_direct_state_zero_checks.py','run_joined_prefill_checks.py','resume_packed_prefill_norm_checks.py','run_fused_head_session_checks.py','test_fused_head_session_reference.py','run_fused_head_session_reference.py','compare_fused_head_argmax.py','compare_vision_global_alpha.py','run_fused_head_argmax_profiles.py','run_vision_global_alpha_profiles.py','run_matched_frame_process.py','compare_vllm_metal.py','test_prefix_reset_session_reference.py')
for name in helpers:
    p=root/'source/test2026/metal'/name;entries[str(p.relative_to(root))]=p
# Preserve validation drivers and model implementations as well as the changed
# source. Never include checkpoint arrays or unrelated workspace locations.
for directory in ('source/xModel','source/src/backends/metal','source/test2026/metal'):
    for p in (root/directory).rglob('*'):
        if p.is_file() and p.suffix in ('.py','.cpp','.mm','.h','.metal'):
            entries[str(p.relative_to(root))]=p
for p in staging.glob('*.json'):entries[str(p.relative_to(root))]=p
libraries={name:digest(root/name/'libgarnet.dylib') for name in ('build-fused-head-argmax','build-global-half-alpha')}
assert libraries['build-global-half-alpha']=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
proof=dict(files={},omitted=omitted,jobs=jobs,original=original,candidate=candidate,library_sha256=libraries,scope='Native full-output gates and ABBA only; no fresh vLLM result')
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(entries.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(entries),bytes=target.stat().st_size,sha256=digest(target),jobs=jobs)))
