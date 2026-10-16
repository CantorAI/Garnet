"""Run exactly two full matched cohorts, archive/audit between them, never transfer while measuring."""
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();tools=Path(__file__).parent;assert out.is_relative_to(root/'logs')
if '--worker' not in sys.argv[3:]:
 out.mkdir(parents=True,exist_ok=False);(out/'controller-source.py').write_bytes(Path(__file__).read_bytes())
 with (out/'output.log').open('w') as stream:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid.json').write_text(json.dumps(dict(pid=p.pid))+'\n');print(json.dumps(dict(pid=p.pid,output=str(out),full_goal_met=False)));raise SystemExit(0)
start=time.monotonic();code=1;phases=[]
def run(name,command):
 (out/'phase.json').write_text(json.dumps(dict(phase=name,command=command,full_goal_met=False),indent=2)+'\n');result=subprocess.run(command);phases.append(dict(phase=name,command=command,exit_code=result.returncode));(out/'phases.json').write_text(json.dumps(phases,indent=2)+'\n');assert result.returncode==0,name
try:
 run('prepare',[sys.executable,str(tools/'prepare_vector_vision_vllm.py'),str(root)])
 for label in ('initial','repeat'):
  trial='vector-vision-vllm-'+label+'-20261016';folder=root/'logs'/trial
  run(label+'-launch',[sys.executable,str(tools/'launch_vector_vision_vllm.py'),str(root),str(folder)])
  pid=json.loads((folder/'pid.json').read_text())['pid'];(out/'phase.json').write_text(json.dumps(dict(phase=label+'-running',pid=pid,folder=str(folder),full_goal_met=False),indent=2)+'\n')
  while True:
   state=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
   if not state.stdout.strip() or state.stdout.strip().startswith('Z'):break
   time.sleep(5)
  assert state.returncode in (0,1) and json.loads((folder/'exit.json').read_text())['exit_code']==0
  archive=root/(trial+'.tar.gz');run(label+'-archive',[sys.executable,str(tools/'archive_vector_vision_vllm.py'),str(root),str(folder),str(archive)])
  evidence=root/(trial+'-archive-audit');evidence.mkdir(exist_ok=False)
  with tarfile.open(archive) as tar:tar.extractall(evidence,filter='data')
  run(label+'-audit',[sys.executable,str(tools/'verify_vector_vision_vllm.py'),str(evidence),str(root/(trial+'-verification.json'))])
 code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start,full_goal_met=False))+'\n')
sys.exit(code)
