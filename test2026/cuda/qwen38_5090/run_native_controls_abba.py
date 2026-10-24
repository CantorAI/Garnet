from pathlib import Path
import hashlib,json,socket,subprocess,sys,time,zipfile
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
out=r/'results/native-controls-abba';assert not out.exists();assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',18082))!=0
library='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b';assert hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()==library
with zipfile.ZipFile(r/'controls-abba-methods.zip') as z:
 m=json.loads(z.read('methods-sha256.json'))
 for n,h in m.items():
  dest=(r/n).resolve();assert dest.is_relative_to(r.resolve()) and dest.parent==r.resolve() and not dest.exists() and hashlib.sha256(z.read(n)).hexdigest()==h
 plan=json.loads(z.read('controls-abba-experiment-plan.json'))
 for n,h in plan['source_sha256'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
 out.mkdir();(out/'plan.json').write_bytes(z.read('controls-abba-experiment-plan.json'))
 for n,h in m.items():(r/n).write_bytes(z.read(n))
finished=[]
def telemetry(label):
 command=['nvidia-smi','--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,power.limit','--format=csv']
 result=subprocess.run(command,capture_output=True,text=True);(out/(label+'-gpu.json')).write_text(json.dumps({'command':command,'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr},indent=2))
for leg in plan['order']:
 prefix=leg['prefix'];name=leg['leg'];print('ABBA_BEGIN',name,flush=True)
 assert hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()==library
 for n,h in plan['source_sha256'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
 telemetry(name+'-before')
 serverlog=(out/(name+'-server-controller.log')).open('wb');server=subprocess.Popen([sys.executable,str(r/('run-'+prefix+'-http-server.py'))],stdout=serverlog,stderr=subprocess.STDOUT)
 # Server handle is authoritative. Poll it during readiness; no timeout restart.
 from urllib.request import urlopen
 deadline=time.monotonic()+1800
 while True:
  assert server.poll() is None,'Server exited during startup; preserve evidence and inspect'
  try:
   with urlopen('http://127.0.0.1:18082/health',timeout=5) as response:health=json.load(response)
   if health.get('native_worker',{}).get('resident_request_controls') is leg['resident_controls']:break
  except Exception:pass
  if time.monotonic()>deadline:raise TimeoutError('Observation deadline; original server may still be live; inspect, never restart')
  time.sleep(5)
 code=run([sys.executable,str(r/('run-'+prefix+'-http-workflow.py'))],r/'results'/(prefix+'-http-controller-ledger'),'correctness','planned-controls-abba-'+name,[r/('run-'+prefix+'-http-workflow.py'),r/('run-'+prefix+'-http-candidate.py'),r/'controls-abba-experiment-plan.json'])
 print('ABBA_WORKFLOW_TERMINAL',name,code,flush=True)
 # Only observed terminal workflow permits identity-guarded owned bridge stop.
 stopped=subprocess.run([sys.executable,str(r/('stop-'+prefix+'-http-worker.py'))]);assert stopped.returncode==0,'Preserve live server and inspect stop failure'
 while server.poll() is None:time.sleep(1)
 serverlog.close();assert server.returncode==1,'Intentional shutdown status changed; inspect original server ledger'
 telemetry(name+'-after')
 assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
 subprocess.run([sys.executable,str(r/('archive-'+prefix+'-http.py'))],check=True)
 archive=r/'results'/(prefix+'-http-evidence.zip');finished.append({'leg':name,'resident_controls':leg['resident_controls'],'workflow_exit_code':code,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()});(out/'progress.json').write_text(json.dumps(finished,indent=2))
 if code:raise SystemExit(code)
 print('ABBA_LEG_ARCHIVED',name,flush=True)
(out/'completed.json').write_text(json.dumps({'status':'four_legs_finished_needs_independent_reaccounting','legs':finished,'scope':plan['scope']},indent=2));print('ABBA_FOUR_LEGS_FINISHED; independent verification required',flush=True)
