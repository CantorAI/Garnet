"""Serial full reproof after parent-only primary telemetry correction."""
from pathlib import Path
import subprocess,sys,json,hashlib,datetime,time
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-native-text-worker';n=Path('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin');s=Path('/Users/shawn/CantorAI/out/research/worldsense-native-text-worker/source');job='gui/501/com.cantorai.cantor'
def run(script,log):
 print('START '+script.name,flush=True)
 with(w/log).open('w')as f:subprocess.run([sys.executable,str(script)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('PASS '+script.name,flush=True)
manifest=json.loads((w/'gateway-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
paths=list(w.glob('*.py'))+[Path(__file__).resolve(),Path(__file__).with_name('ws58_owner_parent.py'),Path(__file__).with_name('ws58_owner_death.py')]
(w/'reproof-driver-source-library-identity.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},libraries={mode:hashlib.sha256((Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-native-text-worker-'+mode)/'libgarnet.dylib').read_bytes()).hexdigest()for mode in ['off','primary','on','fault']},source=manifest),indent=2)+'\n')
run(w/'audit_source.py','source-audit-parent-primary-telemetry.log')
# Failed-model transport cases do not run competing model inference.
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Cantor did not unload')
try:
 for mode in ['on','primary','fault']:
  with(w/('gateway-nomodel-'+mode+'.log')).open('w')as f:subprocess.run([str(n/'xlang3'),'-B',str(w/'gateway_nomodel_guard.py'),mode],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
 print('PASS no-model ownership/error/no-retry guards allthree roles',flush=True)
 (w/'gateway-nomodel-result.json').write_text(json.dumps(dict(all_three_parent_roles_pass=True,scope='Native lifecycle/error plumbing only; no performance result.'),indent=2)+'\n')
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
for script in ['build_corpus.py','build_direct_corpus.py','native_controller.py','text_guard_controller.py','full_correctness.py']:run(w/script,'reproof-'+script.replace('.py','.log'))
# Child lifecycle fault coverage runs only after all numerical checks pass.
run(Path(__file__).with_name('ws58_owner_death.py'),'owner-death-controller.log')
assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
(w/'reproof-complete.json').write_text(json.dumps(dict(full_native_text_VLM_corpus_vision_lifecycle_and_parent_telemetry_reproof_pass=True,source_unchanged=True,no_performance_result=True),indent=2)+'\n')
print('FULL REVISED WS58 CORRECTNESS REPROOF PASS; no performance result',flush=True)
