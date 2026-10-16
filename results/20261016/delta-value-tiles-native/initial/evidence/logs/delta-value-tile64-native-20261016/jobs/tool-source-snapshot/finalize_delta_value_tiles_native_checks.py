"""Watch exact native32 job, archive terminal evidence and publish a gate only after full raw audit."""
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();watched=int(sys.argv[3]);parent=Path(json.loads((root/'preparation.json').read_text())['parent']);tools=Path(__file__).parent
checks=root/'logs/delta-value-tile64-native-checks-20261016';assert json.loads((checks/'pid.json').read_text())['pid']==watched
if '--worker' not in sys.argv[4:]:
 out.mkdir(parents=True,exist_ok=False)
 with (out/'output.log').open('w') as stream:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),str(watched),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid.json').write_text(json.dumps(dict(pid=p.pid))+'\n');print(json.dumps(dict(pid=p.pid,watched_pid=watched,output=str(out))));raise SystemExit(0)
start=time.monotonic();code=1;statuses=[];waits=[]

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def process():
 p=subprocess.run(['/bin/ps','-p',str(watched),'-o','stat=,lstart=,comm='],capture_output=True,text=True);assert p.returncode in (0,1);return p.stdout.strip()

paths=[Path(__file__),tools/'archive_delta_value_tiles_native_checks.py',tools/'verify_delta_value_tiles_native_checks.py',root/'preparation.json',tools/'verify_vector_vision_native_checks.py',tools/'verify_mlp_prefill_native_checks.py'];hashes={str(p):digest(p) for p in paths}
for i,p in enumerate(paths):
 target=out/'controller-source'/str(i)/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
(out/'metadata.json').write_text(json.dumps(dict(watched_pid=watched,watched_process_at_start=process(),source_sha256=hashes,scope=__doc__,native_profiles_launched=False,full_goal_met=False),indent=2)+'\n')

def unchanged():
 for name,sha in hashes.items():assert digest(Path(name))==sha,name

def run(name,command):
 unchanged()
 with (out/(name+'.log')).open('w') as stream:p=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,command=command,exit_code=p.returncode));(out/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n');assert p.returncode==0,name
 unchanged()

try:
 while True:
  state=process()
  if not state or state.split()[0].startswith('Z'):break
  waits.append(dict(elapsed_seconds=time.monotonic()-start,watched_process=state));(out/'waits.json').write_text(json.dumps(waits,indent=2)+'\n')
  if time.monotonic()-start>7200:raise TimeoutError('Observation deadline only; exact native check is neither stopped nor restarted')
  time.sleep(2)
 status=json.loads((checks/'exit.json').read_text());assert isinstance(status['exit_code'],int);(out/'watched-terminal.json').write_text(json.dumps(status)+'\n')
 archive=parent/'delta-value-tile64-native-checks-20261016.tar.gz';assert not archive.exists()
 run('archive',[sys.executable,str(tools/'archive_delta_value_tiles_native_checks.py'),str(root),str(archive)])
 sha=digest(archive);(out/'archive-identity.json').write_text(json.dumps(dict(path=str(archive),sha256=sha,watched_exit=status['exit_code']))+'\n')
 assert status['exit_code']==0,'Native correctness failed; terminal failure archived, no model timings admitted'
 extracted=parent/'validation-audits/delta-value-tile64-native-checks-20261016';extracted.mkdir(parents=True,exist_ok=False)
 with tarfile.open(archive) as tar:
  seen=set()
  for member in tar:
   assert member.isfile() and member.name not in seen;seen.add(member.name);p=(extracted/member.name).resolve();assert p.is_relative_to(extracted);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(member).read())
 run('audit',[sys.executable,str(tools/'verify_delta_value_tiles_native_checks.py'),str(extracted),str(extracted/'verification.json')])
 audit=json.loads((extracted/'verification.json').read_text());assert audit['validation_stages']==32 and audit['delta_value_tile64_trained_steps']==48 and audit['delta_value_tile64_trained_selected_routes']==12
 proof=json.loads((extracted/'evidence/manifest.json').read_text());gate=dict(archive_sha256=sha,full_audit_sha256=digest(extracted/'verification.json'),validation_stages=32,candidate_sha256=proof['library_sha256']['build-delta-value-tile64-prefill'],checks_archive=str(archive),checks_audit_directory=str(extracted),finalizer_folder=str(out),native_profiles_launched=False,full_goal_met=False)
 (root/'delta-value-tile64-native-checks-gate.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate),flush=True);code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start,native_profiles_launched=False))+'\n')
