"""Native XLang3 QoS context/scope contract only; never run with CPython."""
import os,sys,json,time
from pathlib import Path
w=Path(__file__).resolve().parent;sys.path.insert(0,str(w));import qos_probe
assert sys.implementation.name=='xlang3'
port=int(sys.argv[1]);ready=Path(sys.argv[2]);birth=sys.argv[3]
class Service:
 def __init__(self):self.running=True
 def identity(self):return json.dumps(dict(pid=os.getpid(),parent_pid=os.getppid(),birth=birth))
 def report(self):return qos_probe.report()
 def probe(self):return qos_probe.probe()
 def stop(self):self.running=False;return True
service=Service();register_remote_object('qos_context_probe',service);lrpc_listen(port,False)
(w/'native-main-requested-qos.json').write_text(json.dumps(dict(context=json.loads(qos_probe.report()),scope_probe=json.loads(qos_probe.probe())),indent=2)+'\n');ready.write_text(service.identity())
while service.running:time.sleep(.05)
