from pathlib import Path
import subprocess,json,hashlib,datetime,time,sys
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-thread-qos/source';n=r/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor';manifest=json.loads((w/'gateway-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in manifest.items());assert json.loads((w/'gateway-nomodel-result.json').read_text())['all_three_parent_roles_pass']
(w/'text-driver-identities-before.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [w/'text_model_guard.py',w/'invalid_image_loaded_guard.py',Path(__file__).resolve(),n/'xlang3',n/'libxlang3_runtime.dylib']}),indent=2)+'\n')
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Normal service did not unload')
try:
 for mode in ['on','primary','fault']:
  print('START actual native text guard '+mode,flush=True)
  with(w/('text-model-guard-'+mode+'.log')).open('w')as log:subprocess.run([str(n/'xlang3'),'-B',str(w/'text_model_guard.py'),mode],stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
  print('PASS actual native text guard '+mode,flush=True)
 with(w/'invalid-image-loaded-regression.log').open('w')as log:subprocess.run([str(n/'xlang3'),'-B',str(w/'invalid_image_loaded_guard.py')],stdout=log,stderr=subprocess.STDOUT,timeout=300,check=True)
 print('PASS actual native default-C++invalid image and five equivalent loaded-model cases',flush=True)
 assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in manifest.items());(w/'text-model-guards-result.json').write_text(json.dumps(dict(all_three_parent_roles_actual28state_token_prefix_capacity_error_clock_memory_guards_pass=True,source_unchanged=True,no_WorldSense_performance_result=True),indent=2)+'\n')
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
