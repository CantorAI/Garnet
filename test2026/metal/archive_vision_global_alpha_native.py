"""Freeze terminal all-stage vision-alpha validation and full-six inference."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]).resolve();entries={};jobs={}
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def add(p):
 record=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p));entries[record['path']]=(p,record)
for name in ('vision-global-alpha-checks-20261014','vision-global-alpha-native-20261014'):
 folder=root/'logs'/name;status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==0;jobs[name]=status
 for p in sorted(folder.rglob('*')):
  if p.is_file() and p.suffix in ('.log','.json','.txt','.py'):add(p)
for folder in (root/'vision-global-alpha-staging',root/'logs/vision-global-alpha-original-source-20261014'):
 for p in sorted(folder.rglob('*')):
  if p.is_file():add(p)
manifest=json.loads((root/'vision-global-alpha-staging/manifest.json').read_text())
scripts=('run_joined_prefill_checks','run_direct_state_zero_checks','run_global_half_alpha_checks','run_guarded_global_half_alpha_checks','run_guarded_joined_global_mlp_checks','run_guarded_vision_global_alpha_checks','launch_vision_global_alpha_checks','launch_joined_global_mlp_checks','launch_global_half_alpha_checks','launch_quiet_prefill_residual_norm_probe','resume_packed_prefill_norm_checks','compare_vision_global_alpha','run_vision_global_alpha_profiles','queue_vision_global_alpha_profiles','run_matched_frame_process','compare_vllm_metal','test_exact_dense_tail_case_reference','test_exact_dense_tail_reference','reference_dense','archive_vision_global_alpha_native')
for name in set(manifest)|{f'test2026/metal/{n}.py' for n in scripts}:add(root/'source'/name)
checks=json.loads((root/'logs/vision-global-alpha-checks-20261014/jobs/metadata.json').read_text())
for name,sha in checks['source_sha256'].items():assert digest(root/'source'/name)==sha;add(root/'source'/name)
assert all(digest(root/'source'/n)==v for n,v in manifest.items())
candidate=root/'build-vision-global-alpha/libgarnet.dylib';control=root/'build-global-half-alpha/libgarnet.dylib'
assert digest(candidate)==checks['candidate_sha256']=='0997e2ba00ce52766b0e8b1fa0b46f44bcff6b081883e9a8c7ba93c2f3635d23'
assert digest(control)=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
proof=destination.with_suffix('.manifest.json');proof.write_bytes((json.dumps(dict(files=[v for _,v in entries.values()],jobs=jobs,candidate_sha256=digest(candidate),control_sha256=digest(control)),indent=2)+'\n').encode())
with tarfile.open(destination,'w:gz') as archive:
 archive.add(proof,arcname='manifest.json')
 for p,record in entries.values():archive.add(p,arcname=record['path'])
print(json.dumps(dict(files=len(entries),bytes=destination.stat().st_size,sha256=digest(destination))))
