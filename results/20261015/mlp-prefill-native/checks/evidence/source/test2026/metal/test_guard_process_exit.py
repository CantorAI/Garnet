"""A vanished snapshot PID is harmless; a live unknown worker still rejects."""
import ast,json,os,sys
from pathlib import Path
from types import SimpleNamespace
source=Path(__file__).with_name('compare_vllm_metal.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='guard')
class ProcessFixture:
    def __init__(self,live):self.live=live
    def run(self,command,**kwargs):
        assert command==['/bin/ps','-p','700000','-o','comm=']
        return SimpleNamespace(returncode=0 if self.live else 1,stdout='/fixture/xlang3\n' if self.live else '')
    def check_output(self,command,**kwargs):
        assert command==['/bin/ps','-axo','pid=,args='];return ''
checks=[]
for live,verified,reject in [(False,False,False),(True,False,True),(True,True,False)]:
    namespace=dict(json=json,os=os,subprocess=ProcessFixture(live),snapshot=lambda:'700000 0.0 100 /fixture/xlang3',verified_worldsense_service=lambda pid:verified)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'guard-fixture','exec'),namespace)
    rejected=False
    try:namespace['guard']()
    except RuntimeError:rejected=True
    assert rejected==reject
    checks.append(dict(pid_live=live,verified_service=verified,rejected=rejected))
print(json.dumps(dict(process_exit_checks=checks)))
