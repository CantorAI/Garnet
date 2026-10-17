from pathlib import Path
import subprocess,sys,json,hashlib,time,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';source=r/'out/research/worldsense-request-weight-residency/source';job='gui/501/com.cantorai.cantor'
manifest=json.loads((w/'correctness-source-manifest.json').read_text())
def verify_source():assert all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
verify_source()
files=[w/'request_session_guard.py',w/'request_full_session_guard.py',Path(__file__).resolve(),n/'xlang3',r/'out/build/garnet-metal-worldsense-request-weight-residency-on/libgarnet.dylib']
identities={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
(w/'request-guards-dispatch-identities.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=identities),indent=2)+'\n')
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Service unload did not complete')
try:
 for name in ['request_session_guard','request_full_session_guard']:
  print('START '+name,flush=True)
  with(w/(name+'.log')).open('w')as f:subprocess.run([str(n/'xlang3'),'-B',str(w/(name+'.py'))],stdout=f,stderr=subprocess.STDOUT,check=True)
  print('PASS '+name,flush=True)
 verify_source();assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in identities.items())
 (w/'request-guards-result.json').write_text(json.dumps(dict(all_extra_request_trained_guards_pass=True,source_and_dispatched_dependencies_unchanged=True),indent=2)+'\n')
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
