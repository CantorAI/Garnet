"""Wait exact full34 finalizer, freeze it, then run complete sequential native cohorts behind raw gates."""
from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();watched=int(sys.argv[3]);parent=Path(json.loads((root/'preparation.json').read_text())['parent']);tools=Path(__file__).parent
finalizer=parent/'logs/paired-norm-rope-native-finalize-20261016';assert json.loads((finalizer/'pid.json').read_text())['pid']==watched
if '--worker' not in sys.argv[4:]:
 out.mkdir(parents=True,exist_ok=False)
 with (out/'output.log').open('w') as stream:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),str(watched),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)));print(json.dumps(dict(pid=p.pid,watched_finalizer_pid=watched,output=str(out))));raise SystemExit(0)
start=time.monotonic();code=1;statuses=[]
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def state(pid):
 p=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat=,lstart=,comm='],capture_output=True,text=True);assert p.returncode in (0,1);return p.stdout.strip()
tool_ids={p.name:digest(p) for p in tools.iterdir() if p.is_file() and p.suffix in ('.py','.h','.hpp','.cpp','.mm','.metal')}
for name in tool_ids:
 target=out/'controller-source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((tools/name).read_bytes())
(out/'metadata.json').write_text(json.dumps(dict(watched_finalizer_pid=watched,watched_process_at_start=state(watched),tool_source_directory=str(tools),tool_source_sha256=tool_ids,scope=__doc__,full_goal_met=False),indent=2)+'\n')
def unchanged():
 for name,sha in tool_ids.items():assert digest(tools/name)==sha,name
def run(name,command):
 unchanged()
 with (out/(name+'.log')).open('w') as stream:p=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,command=command,exit_code=p.returncode));(out/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n');assert p.returncode==0,name
 unchanged()
def wait_job(folder,pid,label):
 assert json.loads((folder/'pid.json').read_text())['pid']==pid;begin=time.monotonic();waits=[]
 while True:
  current=state(pid)
  if not current or current.split()[0].startswith('Z'):break
  waits.append(dict(elapsed_seconds=time.monotonic()-begin,pid=pid,process=current));(out/(label+'-waits.json')).write_text(json.dumps(waits,indent=2)+'\n')
  if time.monotonic()-begin>7200:raise TimeoutError('Observation deadline only; exact job is not stopped or restarted')
  time.sleep(2)
 status=json.loads((folder/'exit.json').read_text());assert isinstance(status['exit_code'],int);(out/(label+'-terminal.json')).write_text(json.dumps(status)+'\n');return status
def archive_controller(folder,target,status):
 assert folder.is_relative_to(parent) and not target.exists();files={}
 for p in folder.rglob('*'):
  if p.is_file():files['evidence/'+p.relative_to(parent).as_posix()]=p
 gate=root/'paired-norm-rope-native-checks-gate.json'
 if gate.exists():files['evidence/published-checks-gate.json']=gate
 proof=dict(files={},jobs={folder.name:status},scope='Terminal controller commands/source/waits/gate only; numerical/model evidence is separate.',full_goal_met=False)
 with tarfile.open(target,'w:gz') as tar:
  for name,p in sorted(files.items()):
   raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
  raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 return digest(target)
def extract(archive,destination):
 destination.mkdir(parents=True,exist_ok=False)
 with tarfile.open(archive) as tar:
  seen=set()
  for member in tar:
   assert member.isfile() and member.name not in seen;seen.add(member.name);p=(destination/member.name).resolve();assert p.is_relative_to(destination);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(member).read())
 proof=json.loads((destination/'evidence/manifest.json').read_text())
 for name,info in proof['files'].items():
  p=(destination/name).resolve();assert p.is_relative_to(destination) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
 (destination/'archive-verification.json').write_text(json.dumps(dict(archive_sha256=digest(archive),verified_files=len(proof['files']),scope='Archive/manifest bytes only; numerical/model audits separate.',full_goal_met=False),indent=2)+'\n')
try:
 status=wait_job(finalizer,watched,'full34-finalizer');unchanged()
 controller_archive=parent/'paired-norm-rope-native-finalizer-20261016.tar.gz';controller_sha=archive_controller(finalizer,controller_archive,status);(out/'finalizer-archive.json').write_text(json.dumps(dict(path=str(controller_archive),sha256=controller_sha,status=status))+'\n')
 assert status['exit_code']==0,'Native check finalizer failed; terminal controller archived, no model profiles admitted'
 gate=json.loads((root/'paired-norm-rope-native-checks-gate.json').read_text());assert gate['validation_stages']==34 and not gate['native_profiles_launched']
 results=parent/'validation-audits/paired-norm-rope-native-results';results.mkdir(exist_ok=False)
 for label,trial in [('initial','paired-norm-rope-native-20261016'),('repeat','paired-norm-rope-native-confirm-20261016')]:
  folder=root/'logs'/trial
  if label=='repeat':
   assert json.loads((results/'decision-initial.json').read_text())['repeat_admitted'];unchanged()
  run(label+'-launch',[sys.executable,str(tools/'launch_paired_norm_rope_native_profiles.py'),str(root),str(folder),'profiles'])
  if label=='repeat':(folder/'confirmation-launch.json').write_text(json.dumps(dict(initial_decision_sha256=digest(results/'decision-initial.json'),all_sources_unchanged=True,candidate_sha256=gate['candidate_sha256'],scope='One unchanged complete all-six-profile native confirmation; no favorable third repeat.',full_goal_met=False),indent=2)+'\n')
  pid=int(json.loads((folder/'pid.json').read_text())['pid']);status=wait_job(folder,pid,label);unchanged()
  archive=parent/(trial+'.tar.gz')
  if not (folder/'jobs/checks-verification.json').exists():
   sha=archive_controller(folder,archive,status);raise RuntimeError('Profile stopped before runner setup; terminal controller archived '+sha)
  run(label+'-archive',[sys.executable,str(tools/'archive_paired_norm_rope_native_profiles.py'),str(root),str(archive),trial]);assert status['exit_code']==0,'Terminal failed native cohort archived; no interpretation/admission'
  destination=results/label;extract(archive,destination)
  run(label+'-audit',[sys.executable,str(tools/'verify_paired_norm_rope_native_profiles.py'),str(destination),str(destination/'verification.json'),trial])
  args=['initial'] if label=='initial' else ['initial','repeat'];run(label+'-analyze',[sys.executable,str(tools/'analyze_paired_norm_rope_native_results.py'),str(results),*args])
  decision=json.loads((results/('decision-'+label+'.json')).read_text());print(json.dumps(dict(cohort=label,archive_sha256=digest(archive),repeat_admitted=decision['repeat_admitted'],native_confirmation_pass=decision['native_confirmation_pass'],full_goal_met=False)),flush=True)
  if label=='initial' and not decision['repeat_admitted']:break
 code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start,full_goal_met=False))+'\n')
