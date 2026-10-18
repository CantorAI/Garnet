from pathlib import Path
import subprocess,json,hashlib,time,sys
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';s=r/'out/research/worldsense-thread-qos/source';job='gui/501/com.cantorai.cantor'
m=json.loads((w/'gateway-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in m.items())
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Cantor did not unload')
try:
 for mode in ['on','primary','fault']:
  with(w/('gateway-nomodel-'+mode+'.log')).open('w')as log:subprocess.run([str(n/'xlang3'),'-B',str(w/'gateway_nomodel_guard.py'),mode],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
  print('Native worker QoS structured-error/ownership/no-retry/reload/shutdown PASS '+mode,flush=True)
 (w/'gateway-nomodel-result.json').write_text(json.dumps(dict(all_three_parent_roles_pass=True,scoped_QoS_closed_after_model_not_ready_error=True,no_model_inference=True),indent=2)+'\n')
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
