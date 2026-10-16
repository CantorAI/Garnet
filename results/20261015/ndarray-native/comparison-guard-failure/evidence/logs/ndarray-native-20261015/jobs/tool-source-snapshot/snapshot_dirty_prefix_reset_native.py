"""Bounded current-state proof; pending work is never reported as complete."""
import hashlib,json,shlex,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(exist_ok=False)
jobs={}
for name,pid,script,suffix in [('dirty-prefix-reset-checks-20261014',44364,'launch_dirty_prefix_reset_checks.py',['checks','--worker']),('dirty-prefix-reset-native-20261014',45551,'queue_dirty_prefix_reset_profiles.py',['--worker'])]:
 folder=root/'logs'/name;record=dict(pid=pid)
 assert json.loads((folder/'pid.json').read_text())['pid']==pid
 if (folder/'exit.json').exists():record.update(terminal=json.loads((folder/'exit.json').read_text()))
 else:
  command=subprocess.run(['/bin/ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
  assert command.returncode==0 and shlex.split(command.stdout.strip())[1:]==[str(root/'source/test2026/metal'/script),str(root),str(folder),*suffix],'Owned handle missing or changed'
  record['verified_live']=True
 for path in ('pid.json','exit.json','dependency-wait.json','jobs/metadata.json','jobs/statuses.json'):
  p=folder/path
  if p.exists():target=output/name/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
 jobs[name]=record
library_hashes={}
for build,sha in [('build-resident-kv-inplace','73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'),('build-attention-prefix','f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088'),('build-global-half-alpha','887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f')]:
 p=root/build/'libgarnet.dylib';actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==sha;library_hashes[build]=actual
manifest=json.loads((root/'dirty-prefix-reset-staging/manifest.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha
for name in ('manifest.json','original.json'):(output/name).write_bytes((root/'dirty-prefix-reset-staging'/name).read_bytes())
test=root/'build-dirty-prefix-reset/Testing/Temporary/LastTest.log';log=test.read_text();assert 'Dirty prefix reset complete CPU/GPU positive zero, bounds, uninitialized and abort PASS 216' in log
(output/'CTest-LastTest.log').write_bytes(test.read_bytes())
candidate=root/'build-dirty-prefix-reset/libgarnet.dylib';sha=hashlib.sha256(candidate.read_bytes()).hexdigest();assert sha==json.loads((root/'logs/dirty-prefix-reset-checks-20261014/jobs/metadata.json').read_text())['candidate_sha256']
result=dict(scope='Historical bounded snapshot; build/CTest passed, remaining model and serving work requires terminal evidence',jobs=jobs,selected_library_sha256=library_hashes,candidate_sha256=sha,source_files_verified=len(manifest),full_goal_met=False)
(output/'snapshot.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
