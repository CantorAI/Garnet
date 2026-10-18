"""Native XLang3 LRPC feasibility only; no model loading/inference."""
import os,sys,time,json
class Probe:
 def __init__(self,nonce):self.nonce=nonce;self.running=True
 def pid(self):return os.getpid()
 def identity(self):return self.nonce
 def clock(self):return time.monotonic()
 def echo(self,value):return value
 def fail(self):raise ValueError('ws58_expected_error')
 def stop(self):self.running=False;return True
service=Probe(sys.argv[2]);register_remote_object('garnet_text_probe',service);lrpc_listen(int(sys.argv[1]),False)
with open(sys.argv[3],'w')as f:f.write(json.dumps(dict(pid=os.getpid(),native_runtime=True)))
while service.running:time.sleep(.05)
