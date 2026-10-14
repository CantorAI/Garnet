"""Freeze completed global-alpha validation, full outputs and source identities."""
import hashlib,json,platform,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]).resolve();entries={};omitted=[];jobs={}
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def add(p):
 record=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p));entries[record['path']]=(p,record)
for name in ('global-half-alpha-checks-20261014','global-half-alpha-native-20261014'):
 folder=root/'logs'/name;status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==0;jobs[name]=status
 for p in sorted(folder.rglob('*')):
  if not p.is_file():continue
  if p.suffix in ('.log','.json','.txt','.py'):add(p)
  else:omitted.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
for folder in (root/'global-half-alpha-staging',root/'logs/global-half-alpha-original-source-20261014'):
 for p in sorted(folder.rglob('*')):
  if p.is_file():add(p)
manifest=json.loads((root/'global-half-alpha-staging/manifest.json').read_text())
scripts=('run_joined_prefill_checks','run_direct_state_zero_checks','run_global_half_alpha_checks','run_guarded_global_half_alpha_checks','launch_global_half_alpha_checks','launch_quiet_prefill_residual_norm_probe','resume_packed_prefill_norm_checks','compare_global_half_alpha','compare_packed_prefill_norm','compare_constant_head','compare_joined_prefill','run_global_half_alpha_profiles','run_packed_prefill_norm_profiles','run_constant_head_profiles','queue_global_half_alpha_profiles','queue_packed_prefill_norm_profiles','run_matched_frame_process','compare_vllm_metal','test_exact_dense_tail_case_reference','test_exact_dense_tail_reference','reference_dense','archive_global_half_alpha_native')
names=set(manifest)|{f'test2026/metal/{n}.py' for n in scripts}
for name in names:add(root/'source'/name)
checks=json.loads((root/'logs/global-half-alpha-checks-20261014/jobs/metadata.json').read_text())
for name,sha in checks['source_sha256'].items():assert digest(root/'source'/name)==sha;add(root/'source'/name)
assert all(digest(root/'source'/n)==v for n,v in manifest.items())
candidate=root/'build-global-half-alpha/libgarnet.dylib';control=root/'build-attention-prefix/libgarnet.dylib'
assert digest(candidate)=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
assert digest(control)=='f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088'
proof=destination.with_suffix('.manifest.json');proof.write_text(json.dumps(dict(files=[v for _,v in entries.values()],omitted=omitted,jobs=jobs,candidate_sha256=digest(candidate),control_sha256=digest(control),python=sys.version,platform=platform.platform()),indent=2)+'\n')
with tarfile.open(destination,'w:gz') as archive:
 archive.add(proof,arcname='manifest.json')
 for p,record in entries.values():archive.add(p,arcname=record['path'])
print(json.dumps(dict(files=len(entries),bytes=destination.stat().st_size,sha256=digest(destination))))
