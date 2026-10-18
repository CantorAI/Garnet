"""Sequential complete correctness, never a performance comparison."""
from pathlib import Path
import subprocess,sys,json,hashlib,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-thread-qos/source';n=r/'out/build/macos-arm64-release/bin'
assert json.loads((w/'functional-smoke/functional-verification.json').read_text())['actualWorldSense_Galaxy_native_candidate_functional_pass']
m=json.loads((w/'gateway-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in m.items());paths=[p for p in w.glob('*.py')if p.is_file()]
(w/'full-proof-identities-before.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=m,drivers={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},libraries={role:hashlib.sha256((r/('out/build/garnet-metal-worldsense-thread-qos-'+role)/'libgarnet.dylib').read_bytes()).hexdigest()for role in ['off','primary','on','fault']}),indent=2)+'\n')
for script in ['text_guard_controller.py','full_correctness.py','owner_death.py']:
 print('START '+script,flush=True)
 with(w/('proof-'+script.replace('.py','.log'))).open('w')as log:subprocess.run([sys.executable,str(w/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS '+script,flush=True)
assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in m.items());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in json.loads((w/'full-proof-identities-before.json').read_text())['drivers'].items());(w/'full-proof-complete.json').write_text(json.dumps(dict(all_original_native_model_vision_session_and_QoS_typed_scope_parent_death_guards_pass=True,source_drivers_unchanged=True,no_speed_result=True),indent=2)+'\n');print('Full WS59 model/corpus/vision/typedQoS/lifecycle correctness PASS; no speed claim',flush=True)
