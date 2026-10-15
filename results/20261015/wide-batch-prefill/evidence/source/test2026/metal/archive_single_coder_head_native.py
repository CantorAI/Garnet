"""Freeze the failed declaration, repaired full validation and six profiles."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]).resolve();entries={};jobs={}
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def add(p):
 record=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p));entries[record['path']]=(p,record)
for name,expected in [('single-coder-head-checks-20261014',1),('single-coder-head-repair-checks-20261014',0),('single-coder-head-native-20261014',0)]:
 folder=root/'logs'/name;status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==expected;jobs[name]=status
 for p in sorted(folder.rglob('*')):
  if p.is_file() and p.suffix in ('.log','.json','.txt','.py'):add(p)
for name in ('single-coder-head-staging','single-coder-head-repair-staging','logs/single-coder-head-original-source-20261014','logs/single-coder-head-repair-original-source-20261014'):
 for p in sorted((root/name).rglob('*')):
  if p.is_file():add(p)
manifest=json.loads((root/'single-coder-head-repair-staging/manifest.json').read_text())
scripts=('run_joined_prefill_checks','run_direct_state_zero_checks','run_global_half_alpha_checks','run_guarded_global_half_alpha_checks','run_guarded_joined_global_mlp_checks','run_guarded_single_coder_head_checks','run_guarded_single_coder_head_repair_checks','launch_single_coder_head_repair_checks','launch_single_coder_head_checks','launch_joined_global_mlp_checks','launch_global_half_alpha_checks','launch_quiet_prefill_residual_norm_probe','resume_packed_prefill_norm_checks','compare_single_coder_head','run_single_coder_head_profiles','queue_single_coder_head_profiles','run_matched_frame_process','compare_vllm_metal','test_exact_dense_tail_case_reference','test_exact_dense_tail_reference','reference_dense','archive_single_coder_head_native')
for name in set(manifest)|{f'test2026/metal/{n}.py' for n in scripts}:add(root/'source'/name)
checks=json.loads((root/'logs/single-coder-head-repair-checks-20261014/jobs/metadata.json').read_text())
for name,sha in checks['source_sha256'].items():assert digest(root/'source'/name)==sha;add(root/'source'/name)
assert all(digest(root/'source'/n)==v for n,v in manifest.items())
candidate=root/'build-single-coder-head-repaired/libgarnet.dylib';control=root/'build-global-half-alpha/libgarnet.dylib'
assert digest(candidate)==checks['candidate_sha256']=='ec927190e2830e0e8045fd6f18e74a1d90373d21fc6165355b2fe0a16e8abde0'
assert digest(control)=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
proof=destination.with_suffix('.manifest.json');proof.write_bytes((json.dumps(dict(files=[v for _,v in entries.values()],jobs=jobs,candidate_sha256=digest(candidate),control_sha256=digest(control)),indent=2)+'\n').encode())
with tarfile.open(destination,'w:gz') as archive:
 archive.add(proof,arcname='manifest.json')
 for p,record in entries.values():archive.add(p,arcname=record['path'])
print(json.dumps(dict(files=len(entries),bytes=destination.stat().st_size,sha256=digest(destination))))
