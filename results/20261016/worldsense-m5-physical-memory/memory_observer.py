"""OS metadata only. Never reads memory contents or performs inference/images."""
from pathlib import Path
import subprocess,json,hashlib,time,re,threading,shutil,traceback
ROOT=Path('/Users/shawn/CantorAI/out/research/worldsense-physical-memory')
JOB='gui/501/com.cantorai.cantor'
def identity(pid):
 job=subprocess.check_output(['launchctl','print',JOB],text=True);match=re.search(r'^\s*pid = (\d+)',job,re.M);assert match and int(match.group(1))==pid,'Target is not the launchd Cantor PID'
 birth=subprocess.check_output(['ps','-p',str(pid),'-o','lstart=','-o','comm='],text=True).strip();assert 'Cantor' in birth,'Unexpected target executable'
 return {'pid':pid,'birth_and_executable':birth}
class Observer:
 def __init__(self,w,label,pid):
  self.w=Path(w);self.label=label;self.pid=pid;self.target=identity(pid);self.stop_event=threading.Event();self.error=None;self.records=[];self.seen=set();self.thread=threading.Thread(target=self.loop,daemon=True)
 def start(self):self.thread.start()
 def finish(self):
  self.stop_event.set();self.thread.join(35)
  assert not self.thread.is_alive(),'Observer did not drain'
  if self.error:raise RuntimeError(self.error)
  return self.records
 def capture(self,pending,obs):
  key=obs['capture_id'];assert re.fullmatch('[0-9a-f]{64}',key)and pending.name==key+'.pending.json'
  assert obs['pid']==self.pid and identity(self.pid)==self.target,'Worker PID/birth changed'
  folder=self.w/'observations'/self.label/key;folder.mkdir(parents=True,exist_ok=False)
  inv=obs['inventory'];path=Path(inv['path']);assert path.resolve().parent==ROOT/'captures' and path.name==key+'.json'
  raw=path.read_bytes();assert len(raw)==inv['bytes'] and hashlib.sha256(raw).hexdigest()==inv['sha256'];shutil.copy2(path,folder/'native-inventory.json')
  tools={}
  for name,args in [('vmmap',['/usr/bin/vmmap','-summary',str(self.pid)]),('footprint',['/usr/bin/footprint','-p',str(self.pid),'-f','bytes','--swapped','--wired'])]:
   assert identity(self.pid)==self.target
   started=time.monotonic();p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30);elapsed=time.monotonic()-started
   (folder/(name+'.stdout.txt')).write_bytes(p.stdout);(folder/(name+'.stderr.txt')).write_bytes(p.stderr)
   tools[name]={'command':args,'returncode':p.returncode,'observer_seconds':elapsed,'stdout_sha256':hashlib.sha256(p.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(p.stderr).hexdigest()}
   assert p.returncode==0,(name,p.returncode,p.stderr[:120])
  record={'capture_id':key,'worker':self.label,'pid_identity':self.target,'native_boundary':obs,'tools':tools,'scope':'Paused request boundary diagnostic. OS tools may suspend target; all timings excluded from performance claims.'}
  (folder/'observation.json').write_text(json.dumps(record,indent=2));self.records.append(record);(self.w/(self.label+'.observations.json')).write_text(json.dumps(self.records,indent=2))
 def acknowledge(self,key,ok,error=''):
  target=ROOT/'barriers'/(key+'.ack.json');temp=target.with_suffix('.tmp');temp.write_text(json.dumps({'capture_id':key,'ok':ok,'error':error}));temp.replace(target)
 def loop(self):
  try:
   while not self.stop_event.is_set():
    for path in sorted((ROOT/'barriers').glob('*.pending.json')):
     if path.name.removesuffix('.pending.json') in self.seen:continue
     try:obs=json.loads(path.read_text())
     except FileNotFoundError:continue  # Native acknowledgement may atomically rename it.
     key=obs['capture_id']
     if obs['pid']!=self.pid or key in self.seen:continue
     self.seen.add(key)
     try:self.capture(path,obs)
     except Exception as e:
      self.error=traceback.format_exc();self.acknowledge(key,False,str(e));return
     self.acknowledge(key,True)
    self.stop_event.wait(.05)
  except Exception:self.error=traceback.format_exc()
