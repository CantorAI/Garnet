"""Freeze terminal native fusion checks, full outputs and executed sources."""
import hashlib,json,platform,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]).resolve();entries={};omitted=[];jobs={}
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def add(p):
 record=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p));entries[record['path']]=(p,record)
for name in ('residual-norm-checks-20261014','residual-norm-native-20261014'):
 folder=root/'logs'/name;status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==0;jobs[name]=status
 for p in sorted(folder.rglob('*')):
  if not p.is_file():continue
  if p.suffix in ('.log','.json','.txt','.py'):add(p)
  else:omitted.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
for folder in (root/'residual-norm-staging',root/'logs/residual-norm-original-source-20261014'):
 for p in sorted(folder.rglob('*')):
  if p.is_file():add(p)
manifest=json.loads((root/'residual-norm-staging/manifest.json').read_text())
scripts=('run_joined_prefill_checks','run_direct_state_zero_checks','run_residual_norm_checks','launch_residual_norm_checks','compare_residual_norm','compare_constant_head','compare_joined_prefill','run_residual_norm_profiles','run_constant_head_profiles','queue_residual_norm_profiles','run_matched_frame_process','compare_vllm_metal','archive_residual_norm_native')
names=set(manifest)|{f'test2026/metal/{name}.py' for name in scripts}
for name in names:add(root/'source'/name)
checks=json.loads((root/'logs/residual-norm-checks-20261014/jobs/metadata.json').read_text())
for name,sha in checks['source_sha256'].items():assert digest(root/'source'/name)==sha;add(root/'source'/name)
assert all(digest(root/'source'/n)==v for n,v in manifest.items())
candidate=root/'build-residual-norm/libgarnet.dylib';control=root/'build-attention-prefix/libgarnet.dylib'
assert digest(candidate)=='409246a7ab8e069d048177eb490fdf34cd354c8f4e2cb4b718ceead32c49d2ce'
assert digest(control)=='f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088'
proof=destination.with_suffix('.manifest.json');proof.write_text(json.dumps(dict(files=[v for _,v in entries.values()],omitted=omitted,jobs=jobs,candidate_sha256=digest(candidate),control_sha256=digest(control),python=sys.version,platform=platform.platform()),indent=2)+'\n')
with tarfile.open(destination,'w:gz') as archive:
 archive.add(proof,arcname='manifest.json')
 for p,record in entries.values():archive.add(p,arcname=record['path'])
print(json.dumps(dict(files=len(entries),bytes=destination.stat().st_size,sha256=digest(destination))))
