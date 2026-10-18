"""Four fresh owned workers, exactly two original Skill requests per worker."""
from pathlib import Path
import datetime,hashlib,json,os,re,runpy,shutil,subprocess,sys,time
b=Path(__file__).resolve().parent.parent;w=b/'worldsense-paired-gpu-profile';p=w/'capture';r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor'
load=lambda path:json.loads(path.read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
frozen=load(p/'identities-before-dispatch.json');assert not(p/'cohort-manifest.json').exists()
assert all(sha(Path(k))==v for k,v in frozen['external'].items())
for family,m in [('current',frozen['source']),('legacy',frozen['legacy_source'])]:
 root=r/('out/research/worldsense-paired-gpu-profile-'+family)/'source';assert all(sha(root/k)==v for k,v in m.items())
subprocess.run([sys.executable,str(w/'capture_dependencies.py'),'pre-capture'],check=True)
api=runpy.run_path(str(b/'worldsense-http.py'))
def command(args,check=True):return subprocess.run(args,check=check,capture_output=True,text=True)
def stop():
 before=command(['launchctl','print',job],False);match=re.search(r'\n\s+pid = (\d+)',before.stdout);pid=int(match.group(1))if match else None
 command(['sh',str(n/'WorldSense/stop.sh')],False);command(['launchctl','bootout',job],False)
 for _ in range(300):
  job_gone=bool(command(['launchctl','print',job],False).returncode)
  alive=False
  if pid:
   try:os.kill(pid,0);alive=True
   except ProcessLookupError:pass
  if job_gone and not alive:return
  time.sleep(.1)
 raise RuntimeError('Owned Cantor did not stop completely')
def stage(path):
 temporary=n/'libgarnet.dylib.counter-new';shutil.copy2(path,temporary);os.replace(temporary,n/'libgarnet.dylib')
cohort=[];terminal=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),success=False,promotion=False,speed_eligibility=False)
try:
 for worker in frozen['workers']:
  label=worker['label'];assert not(p/(label+'.json')).exists()and not Path(worker['trace']).exists()
  print('START',label,flush=True);stop();stage(Path(worker['library']))
  # A single planned staged-library identity check before this worker starts.
  assert sha(n/'libgarnet.dylib')==worker['library_sha256']
  state=dict(worker,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Reconstructed-source paired observer diagnostic; not uninstrumented speed acceptance.');cohort.append(state);(p/'cohort-manifest.json').write_text(json.dumps(cohort,indent=2)+'\n')
  command(['launchctl','bootstrap','gui/501',worker['plist']]);command([str(n/'xlang3'),'-B',str(w/'wait_cantor.py')])
  with(p/'worldsense-server.log').open('ab')as log:
   subprocess.Popen(['sh',str(n/'WorldSense/start.sh')],env=dict(os.environ,WORLDSENSE_ENABLE_BENCHMARK='1',WORLDSENSE_STORAGE_DIR=str(r/'out/data/worldsense')),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  for _ in range(60):
   try:token=api['login']();break
   except Exception:time.sleep(1)
  else:raise RuntimeError('WorldSense not ready')
  state['cantor_before']=command(['launchctl','print',job]).stdout;state['pid']=int(re.search(r'\n\s+pid = (\d+)',state['cantor_before']).group(1))
  logfile=r/'out/logs/cantor-service.log';errorfile=r/'out/logs/cantor-service-error.log';offset=logfile.stat().st_size;error_offset=errorfile.stat().st_size
  with(p/(label+'.client.log')).open('w')as log:
   result=subprocess.run([str(n/'xlang3'),'-B',str(r/'WorldSense/tools/benchmark_recorded.py'),'http://localhost:9981',str(p/'cases.json'),str(p/(label+'.json')),label],env=dict(os.environ,WORLDSENSE_BENCH_TOKEN=token),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
  state['client_exit']=result.returncode;state['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  stop();state['owned_worker_stopped']=True
  with logfile.open('rb')as stream:stream.seek(offset);(p/(label+'.cantor.log')).write_bytes(stream.read())
  with errorfile.open('rb')as stream:stream.seek(error_offset);(p/(label+'.cantor-stderr.log')).write_bytes(stream.read())
  (p/'cohort-manifest.json').write_text(json.dumps(cohort,indent=2)+'\n');print('FINISH',label,'client exit',result.returncode,flush=True);assert result.returncode==0
 terminal['success']=True
except BaseException as error:terminal['error']=repr(error);raise
finally:
 try:
  stop();stage(w/'before-libgarnet.dylib');subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True);terminal['normal_restored']=True
 finally:
  terminal['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(p/'controller-terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
# Post-terminal copies and full hash verification only after ALL owned workers stop.
for worker in frozen['workers']:
 source=Path(worker['trace']);assert source.is_file();target=p/(worker['label']+'.counters.jsonl');shutil.copy2(source,target);assert sha(source)==sha(target)
assert all(sha(Path(k))==v for k,v in frozen['external'].items())
subprocess.run([sys.executable,str(w/'capture_dependencies.py'),'actual-capture'],check=True)
print('Paired16actualWorldSense calls complete; normalWS10restored; independent audit required',flush=True)
