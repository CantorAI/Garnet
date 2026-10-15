"""Freeze terminal validation, extra session gates and all six native profiles."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();entries={};jobs={}
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def add(p):entries[str(p.relative_to(root))]=(p,dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
for name in ('dirty-prefix-reset-checks-20261014','dirty-prefix-reset-native-20261014'):
 folder=root/'logs'/name;status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==0;jobs[name]=status
 for p in folder.rglob('*'):
  if p.is_file() and p.suffix in ('.log','.json','.txt','.py'):add(p)
gates=root/'logs/dirty-prefix-reset-native-20261014/jobs-session-gates';assert len(json.loads((gates/'statuses.json').read_text()))==3
for p in gates.rglob('*.f32'):add(p)
for folder in (root/'dirty-prefix-reset-staging',root/'logs/dirty-prefix-reset-original-source-20261014'):
 for p in folder.rglob('*'):
  if p.is_file():add(p)
manifest=json.loads((root/'dirty-prefix-reset-staging/manifest.json').read_text())
scripts=('run_joined_prefill_checks','run_direct_state_zero_checks','run_global_half_alpha_checks','run_guarded_global_half_alpha_checks','run_guarded_joined_global_mlp_checks','run_guarded_dirty_prefix_reset_checks','launch_dirty_prefix_reset_checks','launch_vision_global_alpha_checks','launch_joined_global_mlp_checks','launch_global_half_alpha_checks','launch_quiet_prefill_residual_norm_probe','resume_packed_prefill_norm_checks','compare_dirty_prefix_reset','compare_vision_global_alpha','run_dirty_prefix_reset_profiles','run_vision_global_alpha_profiles','queue_dirty_prefix_reset_profiles','queue_vision_global_alpha_profiles','run_matched_frame_process','compare_vllm_metal','test_exact_dense_tail_case_reference','test_exact_dense_tail_reference','reference_dense','run_guarded_prefix_reset_session_tests','run_prefix_reset_session_reference','test_prefix_reset_session_reference','archive_dirty_prefix_reset_native')
for name in set(manifest)|{f'test2026/metal/{n}.py' for n in scripts}:add(root/'source'/name)
checks=json.loads((root/'logs/dirty-prefix-reset-checks-20261014/jobs/metadata.json').read_text())
for name,sha in checks['source_sha256'].items():assert digest(root/'source'/name)==sha;add(root/'source'/name)
assert all(digest(root/'source'/n)==sha for n,sha in manifest.items())
candidate=root/'build-dirty-prefix-reset/libgarnet.dylib';control=root/'build-global-half-alpha/libgarnet.dylib';assert digest(candidate)==checks['candidate_sha256']
assert digest(control)=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):add(root/'models'/model/'config.json')
proof=target.with_suffix('.manifest.json');proof.write_bytes((json.dumps(dict(files=[v for _,v in entries.values()],jobs=jobs,candidate_sha256=digest(candidate),control_sha256=digest(control)),indent=2)+'\n').encode())
with tarfile.open(target,'w:gz') as archive:
 archive.add(proof,arcname='manifest.json')
 for p,info in entries.values():archive.add(p,arcname=info['path'])
print(json.dumps(dict(files=len(entries),bytes=target.stat().st_size,sha256=digest(target))))
