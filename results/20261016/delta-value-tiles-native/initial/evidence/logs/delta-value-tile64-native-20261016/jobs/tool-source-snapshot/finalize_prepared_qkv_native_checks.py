"""Watch one exact native check job, archive terminal evidence, audit success; never start inference."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();watched=int(sys.argv[3]);worker='--worker' in sys.argv[4:]
folder=root/'logs/prepared-qkv-native-checks-20261016';parent=Path(json.loads((root/'preparation.json').read_text())['parent']);assert json.loads((folder/'pid.json').read_text())['pid']==watched
if not worker:
 out.mkdir(parents=True,exist_ok=False)
 with (out/'output.log').open('w') as stream:process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),str(watched),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid.json').write_text(json.dumps(dict(pid=process.pid,watched_pid=watched)));print(json.dumps(dict(pid=process.pid,watched_pid=watched,output=str(out))));sys.exit(0)
start=time.monotonic();code=1;statuses=[]
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def live():
 result=subprocess.run(['ps','-p',str(watched),'-o','stat='],capture_output=True,text=True);value=result.stdout.strip();return bool(value and not value.startswith('Z'))
def run(name,command):
 with (out/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,command=command,exit_code=result.returncode));(out/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n');assert result.returncode==0,name
try:
 paths=[Path(__file__),Path(__file__).with_name('archive_prepared_qkv_native_checks.py'),Path(__file__).with_name('verify_prepared_qkv_native_checks.py'),parent/'work/freeze_mlp_evidence.py'];hashes={p.as_posix():digest(p) for p in paths}
 (out/'metadata.json').write_text(json.dumps(dict(watched_pid=watched,source_sha256=hashes,scope=__doc__,full_goal_met=False),indent=2)+'\n')
 while not (folder/'exit.json').exists():
  assert live(),'Watched launcher missing without terminal status; do not restart or mutate source'
  if time.monotonic()-start>9000:raise TimeoutError('Exact native check watcher exceeded deadline; no restart')
  time.sleep(3)
 status=json.loads((folder/'exit.json').read_text());deadline=time.monotonic()+30
 while live() and time.monotonic()<deadline:time.sleep(.2)
 assert not live(),'Terminal status observed but owned launcher not yet gone'
 archive=parent/'prepared-qkv-native-checks-20261016.tar.gz'
 run('archive',[sys.executable,str(Path(__file__).with_name('archive_prepared_qkv_native_checks.py')),str(root),str(archive)])
 sha=digest(archive);(out/'terminal-archive.json').write_text(json.dumps(dict(archive=str(archive),sha256=sha,status=status,full_goal_met=False),indent=2)+'\n')
 assert status['exit_code']==0,'Native checks failed; terminal archive preserved, no performance launch'
 extracted=parent/'validation-audits/prepared-qkv-native-checks-20261016'
 run('extract',[sys.executable,str(parent/'work/freeze_mlp_evidence.py'),str(archive),str(extracted),sha])
 run('verify',[sys.executable,str(Path(__file__).with_name('verify_prepared_qkv_native_checks.py')),str(extracted),str(extracted/'verification.json')])
 (root/'prepared-qkv-native-checks-archive-verification.json').write_bytes((extracted/'verification.json').read_bytes())
 for name,sha in hashes.items():assert digest(Path(name))==sha
 print('Terminal native evidence archived and full33 audit passed; no native performance or overall goal claim.',flush=True);code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start)))
sys.exit(code)
