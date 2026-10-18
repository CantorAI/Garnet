"""Run original full proof and actual new fault boundary, then restore WS10."""
from pathlib import Path
import subprocess,sys,json,time,os,hashlib,datetime
w=Path(__file__).resolve().parent.parent/'worldsense-gpu-counter-profile';r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor'
assert(w/'learned-identities-before.json').is_file()
subprocess.run([sys.executable,str(w/'full_correctness.py')],check=True)
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Service did not unload')
try:
 with(w/'counter-fault-model-recovery.log').open('w')as log:subprocess.run([str(n/'xlang3'),'-B',str(w/'counter_fault_model_guard.py')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300,env=dict(os.environ,GARNET_METAL_COUNTER_TRACE='1',GARNET_METAL_COUNTER_TRACE_FILE=str(w/'counter-fault-model-recovery.counters.jsonl')))
 print('Actual counter fault model recovery PASS',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
(w/'learned-controller-result.json').write_text(json.dumps(dict(full_original_learned_numerical_and_counter_fault_model_proof_pass=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_speed_result=True),indent=2)+'\n')
