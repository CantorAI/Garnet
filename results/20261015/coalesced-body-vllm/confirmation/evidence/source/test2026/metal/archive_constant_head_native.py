"""Freeze completed constant-head native checks, outputs and executed sources."""
import hashlib,json,platform,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]).resolve();entries={};omitted=[];jobs={}
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def add(p):
 record=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p));entries[record['path']]=(p,record)
for name in ('constant-head-checks-20261014','constant-head-native-20261014'):
 folder=root/'logs'/name;status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==0;jobs[name]=status
 for p in sorted(folder.rglob('*')):
  if not p.is_file():continue
  if p.suffix in ('.log','.json','.txt','.py'):add(p)
  else:omitted.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
for folder in (root/'constant-head-staging',root/'logs/constant-head-original-source-20261014'):
 for p in sorted(folder.rglob('*')):
  if p.is_file():add(p)
manifest=json.loads((root/'constant-head-staging/manifest.json').read_text())
names=set(manifest)|{f'test2026/metal/{name}.py' for name in ('run_joined_prefill_checks','run_direct_state_zero_checks','launch_constant_head_checks','compare_constant_head','compare_direct_state_zero','compare_joined_prefill','run_constant_head_profiles','run_joined_prefill_profiles','queue_constant_head_profiles','run_matched_frame_process','compare_vllm_metal')}
for name in names:add(root/'source'/name)
assert all(digest(root/'source'/n)==v for n,v in manifest.items())
candidate=root/'build-constant-bf16-head/libgarnet.dylib';control=root/'build-attention-prefix/libgarnet.dylib'
assert digest(candidate)=='df1c1912b2daf40c86925d323747d60db6a4950723ac04562f06854da1c7aaa3'
assert digest(control)=='f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088'
proof=destination.with_suffix('.manifest.json');proof.write_text(json.dumps(dict(files=[v for _,v in entries.values()],omitted=omitted,jobs=jobs,candidate_sha256=digest(candidate),control_sha256=digest(control),python=sys.version,platform=platform.platform()),indent=2)+'\n')
with tarfile.open(destination,'w:gz') as archive:
 archive.add(proof,arcname='manifest.json')
 for p,record in entries.values():archive.add(p,arcname=record['path'])
print(json.dumps(dict(files=len(entries),bytes=destination.stat().st_size,sha256=digest(destination))))
