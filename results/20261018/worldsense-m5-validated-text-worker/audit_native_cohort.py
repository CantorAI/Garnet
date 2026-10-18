"""Extra WS61 stable-worker/no-hidden-call resource admission, before metric audit."""
from pathlib import Path
import json,hashlib,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');environment=json.loads((w/'environment.json').read_text());manifest={x['label']:x for x in json.loads((w/'cohort-manifest.json').read_text())};rows=[]
for label in ['control-1','candidate-1']:
 d=json.loads((w/(label+'.json')).read_text());calls=[c for item in d['items']for c in item['result']['model_response'].values()if isinstance(c,dict)and c.get('status')=='ok'];assert len(calls)==4
 if label.startswith('control'):
  assert all('native_gateway'not in c['timings']for c in calls);continue
 calls.sort(key=lambda c:c['timings']['native_gateway']['first_token_monotonic']);parent=None;owned=None;last_count=0;seen_remote=0
 for c in calls:
  g=c['timings']['native_gateway'];pid=g['parent_pid'];assert parent in [None,pid];parent=pid;child=g['child']
  if child is None:assert owned is None;continue
  identity=(child['support']['pid'],child['birth_id'],child['support']['library_path']);assert owned in [None,identity];owned=identity
  count=child['infer_calls'];remote=g['route']=='native_lrpc_raw_text';assert count==last_count+int(remote);last_count=count;seen_remote+=int(remote)
  rows.append(dict(worker=label,model=c['model_id'],parent_pid=pid,child_pid=identity[0],birth_id=identity[1],native_worker_call_index=count,remote=remote,cohort_process_count=c['timings']['memory_after_decode']['cohort_process_count']))
 assert owned is not None and seen_remote==2 and last_count==2
for mode,sha in environment['libraries'].items():
 p=r/('out/build/garnet-metal-worldsense-validated-text-worker-'+mode)/'libgarnet.dylib';assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
assert hashlib.sha256((r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib').read_bytes()).hexdigest()==environment['legacy_backend']['sha256']
for role,value in environment['native_hosts'].items():
 assert hashlib.sha256((r/('out/build/garnet-metal-worldsense-validated-text-worker-'+role)/'libgarnet_worker_host.dylib').read_bytes()).hexdigest()==value
(w/'cohort-native-identity-audit.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),resource_scope_only=True,stable_parent_child_birth_per_worker=True,exact_two_child_calls_no_hidden_retry=True,all_three_build_libraries_unchanged=True,records=rows),indent=2)+'\n');print('Actual resource cohort stable owned-child identity, no hidden calls and frozen RAW child library PASS')
