"""Wait exact full33 finalizer; archive it, then complete native cohorts with immutable evidence gates."""
from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();watched=int(sys.argv[3]);parent=Path(json.loads((root/'preparation.json').read_text())['parent']);finalizer=parent/'logs/prepared-qkv-native-finalize-countfix-20261016';tools=Path(__file__).parent
assert json.loads((finalizer/'pid.json').read_text())['pid']==watched
if '--worker' not in sys.argv[4:]:
 out.mkdir(parents=True,exist_ok=False)
 with (out/'output.log').open('w') as stream:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),str(watched),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid.json').write_text(json.dumps(dict(pid=p.pid,watched_finalizer=watched)));print(json.dumps(dict(pid=p.pid,watched_finalizer=watched,output=str(out))));sys.exit(0)
start=time.monotonic();code=1;statuses=[]
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def live(pid):
 result=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True);state=result.stdout.strip();return bool(state and not state.startswith('Z'))
def wait(folder,pid):
 while not (folder/'exit.json').exists():
  assert live(pid),'Exact watched process missing without terminal status; no restart'
  if time.monotonic()-start>14400:raise TimeoutError('Native continuation deadline; no restart')
  time.sleep(3)
 status=json.loads((folder/'exit.json').read_text());deadline=time.monotonic()+30
 while live(pid) and time.monotonic()<deadline:time.sleep(.2)
 assert not live(pid),'Terminal file observed but exact process still live'
 return status
def run(name,command):
 with (out/(name+'.log')).open('w') as stream:p=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,command=command,exit_code=p.returncode));(out/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n');assert p.returncode==0,name
def archive_finalizer(status):
 target=parent/'prepared-qkv-native-finalizer-countfix-20261016.tar.gz';assert not target.exists();files={}
 for p in finalizer.rglob('*'):
  if p.is_file():files['evidence/'+p.relative_to(parent).as_posix()]=p
 for p in tools.glob('*prepared_qkv_native_checks*.py'):files['evidence/tools/'+p.name]=p
 gate=root/'prepared-qkv-native-checks-archive-verification.json'
 if gate.exists():files['evidence/published-checks-gate.json']=gate
 proof=dict(files={},jobs={finalizer.name:status},scope='Terminal full33 finalizer commands/source/gate only; actual numerical archive remains separate.',full_goal_met=False)
 with tarfile.open(target,'w:gz') as tar:
  for name,p in sorted(files.items()):
   raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
  raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 (out/'finalizer-archive.json').write_text(json.dumps(dict(path=str(target),sha256=digest(target),status=status),indent=2)+'\n')
try:
 names=['continue_prepared_qkv_native.py','launch_prepared_qkv_native_profiles.py','run_prepared_qkv_native_profiles.py','compare_prepared_qkv_native.py','archive_prepared_qkv_native_profiles.py','verify_prepared_qkv_native_profiles.py','analyze_prepared_qkv_native_results.py']
 identities={name:digest(tools/name) for name in names}
 for name in names:
  target=out/'source-snapshot'/name;target.parent.mkdir(exist_ok=True);target.write_bytes((tools/name).read_bytes())
 (out/'metadata.json').write_text(json.dumps(dict(watched_finalizer=watched,source_sha256=identities,scope=__doc__,full_goal_met=False),indent=2)+'\n')
 status=wait(finalizer,watched);archive_finalizer(status);assert status['exit_code']==0,'Finalizer failed; preserve archive and stop before native timing'
 assert json.loads((root/'prepared-qkv-native-checks-archive-verification.json').read_text())['validation_stages']==33
 analysis=parent/'validation-audits/prepared-qkv-native-comparison-20261016';analysis.mkdir(parents=True,exist_ok=False)
 for label,trial in [('initial','prepared-qkv-native-20261016'),('repeat','prepared-qkv-native-confirm-20261016')]:
  folder=root/'logs'/trial;run(label+'-launch',[sys.executable,str(tools/'launch_prepared_qkv_native_profiles.py'),str(root),str(folder),'profiles'])
  pid=json.loads((folder/'pid.json').read_text())['pid'];(out/(label+'-launch.json')).write_text(json.dumps(dict(pid=pid,folder=str(folder)),indent=2)+'\n')
  status=wait(folder,pid)
  archive=parent/(trial+'.tar.gz');run(label+'-archive',[sys.executable,str(tools/'archive_prepared_qkv_native_profiles.py'),str(root),str(archive),trial])
  sha=digest(archive);(out/(label+'-archive.json')).write_text(json.dumps(dict(path=str(archive),sha256=sha,status=status),indent=2)+'\n');assert status['exit_code']==0,'Native execution failed; terminal raw archive preserved'
  extracted=analysis/label;run(label+'-extract',[sys.executable,str(parent/'work/freeze_mlp_evidence.py'),str(archive),str(extracted),sha])
  run(label+'-verify',[sys.executable,str(tools/'verify_prepared_qkv_native_profiles.py'),str(extracted),str(extracted/'verification.json'),trial])
  command=[sys.executable,str(tools/'analyze_prepared_qkv_native_results.py'),str(analysis),'initial']
  if label=='repeat':command.append('repeat')
  run(label+'-decision',command);decision=json.loads((analysis/('decision-'+label+'.json')).read_text())
  for name,sha in identities.items():assert digest(tools/name)==sha
  if label=='initial' and not decision['repeat_admitted']:
   print('Complete initial native screen failed; no favorable repeat or integration.',flush=True);break
 print('Native comparison continuation terminal; all raw cohorts archived/audited, no fresh vLLM or full-goal claim.',flush=True);code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start)))
sys.exit(code)
