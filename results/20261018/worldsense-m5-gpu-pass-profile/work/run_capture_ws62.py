"""One fresh packed-profile worker, two original fixed-frame Skill requests."""
from pathlib import Path
import os,sys,time,json,hashlib,subprocess,shutil,runpy,traceback,datetime
base=Path(__file__).resolve().parent.parent;w=base/'worldsense-gpu-counter-profile';p=w/'capture';r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor'
frozen=json.loads((p/'identities-before-dispatch.json').read_text());assert not(p/'candidate-1.json').exists()and not Path(frozen['capture_trace_path']).exists()
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert all(sha(Path(k))==v for k,v in frozen['external'].items())
subprocess.run([sys.executable,str(w/'dependency_closure.py'),'pre-capture'],check=True)
api=runpy.run_path(str(base/'worldsense-http.py'))
def command(args,check=True):return subprocess.run(args,check=check,capture_output=True,text=True)
def stop():
 command(['sh',str(n/'WorldSense/stop.sh')],False);command(['launchctl','bootout',job],False)
 for _ in range(200):
  if command(['launchctl','print',job],False).returncode:return
  time.sleep(.1)
 raise RuntimeError('Cantor failed to unload')
def stage(path):
 temporary=n/'libgarnet.dylib.counter-new';shutil.copy2(path,temporary);os.replace(temporary,n/'libgarnet.dylib')
state=dict(label='candidate-1',scope='Diagnostic only. Four exact actual-pipeline calls; no speed comparison/promotion.',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
try:
 print('START bounded actualWorldSense counter diagnostic',flush=True);stop()
 library=r/'out/build/garnet-metal-worldsense-gpu-counter-profile-on/libgarnet.dylib';stage(library)
 state.update(library_sha256=sha(library),serving_sha256=sha(r/'out/research/worldsense-gpu-counter-profile/source/runtime/metal_serving.py'),counter_file=frozen['capture_trace_path'])
 command(['launchctl','bootstrap','gui/501',frozen['capture_plist']]);command([str(n/'xlang3'),'-B',str(w/'wait_cantor.py')])
 with(p/'worldsense-server.log').open('ab')as log:subprocess.Popen(['sh',str(n/'WorldSense/start.sh')],env=dict(os.environ,WORLDSENSE_ENABLE_BENCHMARK='1',WORLDSENSE_STORAGE_DIR=str(r/'out/data/worldsense')),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(60):
  try:token=api['login']();break
  except Exception:time.sleep(1)
 else:raise RuntimeError('WorldSense not ready')
 state['cantor_before']=command(['launchctl','print',job]).stdout
 logfile=r/'out/logs/cantor-service.log';errorfile=r/'out/logs/cantor-service-error.log';offset=logfile.stat().st_size;error_offset=errorfile.stat().st_size
 with(p/'candidate-1.client.log').open('w')as log:result=subprocess.run([str(n/'xlang3'),'-B',str(r/'WorldSense/tools/benchmark_recorded.py'),'http://localhost:9981',str(p/'cases.json'),str(p/'candidate-1.json'),'candidate-1'],env=dict(os.environ,WORLDSENSE_BENCH_TOKEN=token),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 state['client_exit']=result.returncode
 with logfile.open('rb')as f:f.seek(offset);(p/'candidate-1.cantor.log').write_bytes(f.read())
 with errorfile.open('rb')as f:f.seek(error_offset);(p/'candidate-1.cantor-stderr.log').write_bytes(f.read())
 (p/'cohort-manifest.json').write_text(json.dumps([state],indent=2)+'\n');print('FINISH client exit '+str(result.returncode),flush=True)
 assert result.returncode==0
finally:
 stop();stage(w/'before-libgarnet.dylib');subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
 print('RESTORED selected WS10/fourfiles/noCam/benchmark404',flush=True)
# File copy/hash only after the measured/diagnostic process is stopped.
trace=Path(frozen['capture_trace_path']);assert trace.is_file();shutil.copy2(trace,p/'candidate-1.counters.jsonl');assert sha(trace)==sha(p/'candidate-1.counters.jsonl')
subprocess.run([sys.executable,str(w/'dependency_closure.py'),'actual-counter-capture'],check=True)
