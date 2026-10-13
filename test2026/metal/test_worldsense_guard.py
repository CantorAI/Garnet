"""Read-only guard regression checks using a live inspected WorldSense launcher.

Arguments: comparison script, new evidence directory, live WorldSense PID.
No inline process code is executed or saved; only its fingerprint is retained.
"""
import ast,hashlib,json,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace

script=Path(sys.argv[1]);output=Path(sys.argv[2]);pid=int(sys.argv[3]);output.mkdir(parents=True,exist_ok=False)
tree=ast.parse(script.read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verified_worldsense_service')
scope=dict(subprocess=subprocess,hashlib=hashlib,json=json,time=time,background_checks=[],output=output,
 worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py'))
exec(compile(ast.Module(body=[function],type_ignores=[]),str(script),'exec'),scope)
verify=scope['verified_worldsense_service'];service=scope['worldsense_script']
actual=subprocess.run(['/bin/ps','-p',str(pid),'-o','ppid=,args='],capture_output=True,text=True)
assert actual.returncode==0,'Live service disappeared; do not fabricate a positive guard result'
parts=actual.stdout.strip().split(None,1);assert len(parts)==2
identity=parts[1];results=[]

class Probe:
 def __init__(self,command=identity,libraries='',ps_code=0,lsof_code=0):
  self.command=command;self.libraries=libraries;self.ps_code=ps_code;self.lsof_code=lsof_code
 def run(self,command,**kwargs):
  if command[0]=='/bin/ps':return SimpleNamespace(returncode=self.ps_code,stdout='1 '+self.command)
  assert command[0]=='/usr/sbin/lsof'
  return SimpleNamespace(returncode=self.lsof_code,stdout=self.libraries)

def check(label,expected,probe):
 scope['subprocess']=probe
 observed=verify(pid);assert observed==expected,(label,observed,expected)
 results.append(dict(name=label,passed=True,expected=expected))

check('exact inspected inline launcher',True,Probe())
check('same name with altered code',False,Probe(command=identity+';import mlx'))
check('WorldSense name alone',False,Probe(command=str(service.parents[2]/'xlang3')+' -B -u -c WorldSense'))
check('wrong executable',False,Probe(command=identity.replace(str(service.parents[2]/'xlang3'),'/tmp/xlang3',1)))
check('missing PID',False,Probe(ps_code=1))
check('unreadable loaded libraries',False,Probe(lsof_code=1))
for library in ('libgarnet','libmlx','libtorch','Metal.framework','MetalPerformanceShaders.framework',
 'libopenvino','CoreML.framework','AppleNeuralEngine','ANE.framework','BNNS'):
 check('reject '+library,False,Probe(libraries='n/private/'+library))

class ChangedLaunch:
 def is_file(self):return True
 def read_bytes(self):return b'# changed entrypoint, even without inference keywords\n'

class ChangedSource:
 parents=service.parents
 def with_name(self,name):assert name=='launch.py';return ChangedLaunch()

scope['worldsense_script']=ChangedSource()
check('changed launcher source',False,Probe())
scope['worldsense_script']=service;scope['subprocess']=subprocess
supervisor_bytes=service.read_bytes()
assert hashlib.sha256(supervisor_bytes).hexdigest()=='b904bcf8ece0b74dc160a95aa833b58c8d33536ba90f924465a61b95e61238ed','Supervisor changed; inspect it before extending the guard'
class SupervisorSource:
 parents=service.parents
 def __init__(self,data):self.data=data
 def __str__(self):return str(service)
 def is_file(self):return True
 def read_bytes(self):return self.data
 def read_text(self,**kwargs):return self.data.decode()
scope['worldsense_script']=SupervisorSource(supervisor_bytes)
supervisor_command=str(service.parents[2]/'xlang3')+' -B -u '+str(service)
check('pinned forwarding-only supervisor',True,Probe(command=supervisor_command))
check('supervisor with loaded inference library',False,Probe(command=supervisor_command,libraries='n/private/libgarnet'))
scope['worldsense_script']=SupervisorSource(supervisor_bytes+b'\n')
check('changed supervisor with forwarding marker',False,Probe(command=supervisor_command))
scope['worldsense_script']=service;scope['subprocess']=subprocess
assert verify(pid),'Current live service did not pass the inspected-source/library checks'
results.append(dict(name='actual live read-only verification',passed=True,expected=True))
summary=dict(scope='Guard identity/rejection checks; no GPU inference',checks=results,
 comparison_source_sha256=hashlib.file_digest(script.open('rb'),'sha256').hexdigest(),
 live_pid=pid,all_passed=True)
(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('WorldSense guard checks PASS',len(results),flush=True)
