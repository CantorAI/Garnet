"""Owned launcher for native watchdog probe; no model or active service changes."""
from pathlib import Path
import subprocess,os,json,time,sys
w=Path(__file__).resolve().parent;folder=Path(sys.argv[1]);nonce=sys.argv[2];ready=folder/'native-ready.json';env=dict(os.environ,GARNET_NATIVE_WORKER_PARENT=str(os.getpid()))
with(folder/'native.log').open('w')as log:
 child=subprocess.Popen(['/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/xlang3','-B',str(w/'native_host_guard.py'),'watch',str(ready),nonce,str(os.getpid())],env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(200):
   assert child.poll()is None,'Owned native probe exited before readiness'
   if ready.exists():break
   time.sleep(.05)
  else:raise RuntimeError('Native probe readiness timeout')
  (folder/'parent-ready.json').write_text(json.dumps(dict(parent_pid=os.getpid(),child_pid=child.pid,nonce=nonce))+'\n')
  for _ in range(200):
   if(folder/'exit.request').exists():os._exit(0)
   assert child.poll()is None
   time.sleep(.05)
  raise RuntimeError('Owner-exit request timeout')
 finally:
  if child.poll()is None:
   child.terminate()
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:child.kill();child.wait()
