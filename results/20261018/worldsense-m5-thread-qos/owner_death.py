"""Owned native parent-death proofs, independent of throughput measurements."""
from pathlib import Path
import subprocess,json,os,time,hashlib,sys,datetime,traceback
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-thread-qos';n=Path('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin');driver=Path(__file__).with_name('owner_parent.py');job='gui/501/com.cantorai.cantor';records=[];identity=w/'owner-death-driver-identity.json'
identity.write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),drivers={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [driver,Path(__file__).resolve()]}),indent=2)+'\n')
def ps(pid):
 p=subprocess.run(['/bin/ps','-p',str(pid),'-o','pid=,ppid=,stat=,lstart=,comm='],capture_output=True,text=True,check=False);return p.stdout.strip()if p.returncode==0 else''
def wait_file(path,proc,seconds):
 deadline=time.monotonic()+seconds
 while not path.exists():
  assert proc.poll()is None,'Native parent terminated before expected marker'
  assert time.monotonic()<deadline,'Native parent marker deadline: '+path.name
  time.sleep(.02)
def alive_same_child(pid,expected_start):
 value=ps(pid)
 if not value:return False
 parts=value.split();assert parts[3:8]==expected_start,'PID reused after owned child exit';return not parts[2].startswith('Z')
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Cantor service did not unload')
try:
 for case in ['bootstrap','loaded_idle','active_infer']:
  folder=w/('owner-death-'+case);folder.mkdir(exist_ok=False);proc=None;child=None;start=None
  try:
   with(folder/'parent.log').open('w')as log:
    proc=subprocess.Popen([str(n/'xlang3'),'-B',str(driver),case,str(folder)],stdout=log,stderr=subprocess.STDOUT)
    wait_file(folder/'ready.json',proc,180);ready=json.loads((folder/'ready.json').read_text());assert ready['parent_pid']==proc.pid;info=ready['identity'];assert info['parent_pid']==proc.pid;owned=info['children'][0];child=owned['support']['pid'];assert owned['support']['parent_pid']==proc.pid and owned['support']['watchdog_active']
    before=ps(child);parts=before.split();assert int(parts[0])==child and int(parts[1])==proc.pid and parts[-1].endswith('/xlang3');start=parts[3:8]
    (folder/'start.request').write_text('start\n');samples=[]
    if case=='active_infer':
     wait_file(folder/'infer-dispatched.json',proc,5);confirmed=False
     for j in range(3):
      sample=folder/('owned-child-active-stack-'+str(j)+'.txt');result=subprocess.run(['/usr/bin/sample',str(child),'1','1','-mayDie','-file',str(sample)],capture_output=True,text=True,timeout=15);text=sample.read_text()if sample.exists()else''
      matches=[line.strip()for line in text.splitlines()if any(x in line for x in ['MetalDecoderSession::','MetalCompiledGraph::Execute','MetalCompiledGraph::execute'])]
      samples.append(dict(exit=result.returncode,log=result.stdout+result.stderr,stack_matches=matches));assert result.returncode==0,'Unable to sample owned child'
      if matches:confirmed=True;break
     (folder/'active-stack-verification.json').write_text(json.dumps(dict(attempts=samples,actual_native_inference_stack_seen=confirmed),indent=2));assert confirmed,'No active native Garnet inference stack evidence';assert not(folder/'unexpected-infer-completed.json').exists(),'Inference completed before owner-death injection'
    assert proc.poll()is None and alive_same_child(child,start);current=ps(child);assert int(current.split()[1])==proc.pid
    killed=time.monotonic();proc.kill();assert proc.wait(timeout=10)<0
    deadline=killed+5
    while alive_same_child(child,start):
     assert time.monotonic()<deadline,'Native watchdog left owned child alive after parent death'
     time.sleep(.02)
    row=dict(case=case,parent_exit=proc.returncode,parent_pid=proc.pid,child_pid=child,birth_id=owned['birth_id'],child_before=before,child_after=ps(child),observed_not_running_after_seconds=time.monotonic()-killed,active_native_stack_required=case=='active_infer',scope='Observed watchdog cleanup for owned child; orphan exit status unavailable after reparenting. Correctness injection only, not throughput.');records.append(row);(w/'owner-death-progress.json').write_text(json.dumps(records,indent=2));print('Owned native watchdog parent-death PASS '+case,flush=True)
  finally:
   if proc is not None and proc.poll()is None:proc.kill();proc.wait(timeout=10)
   # On a failed watchdog assertion, clean up only the still-matching owned child.
   if child is not None and start is not None and alive_same_child(child,start):
    os.kill(child,15);deadline=time.monotonic()+2
    while alive_same_child(child,start)and time.monotonic()<deadline:time.sleep(.02)
    if alive_same_child(child,start):os.kill(child,9)
 (w/'owner-death-result.json').write_text(json.dumps(dict(all_three_parent_death_cases_pass=True,records=records),indent=2)+'\n')
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
