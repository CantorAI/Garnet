"""Orchestration only: native C++ and native XLang3 endpoints, no inference."""
from pathlib import Path
import subprocess,sys,os,json,time,uuid,hashlib,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';exe=w/'feasibility-client'
command=['/usr/bin/c++','-std=c++17','-O2','-I',str(r/'xlang3/sdk'),str(w/'feasibility_client.cpp'),str(n/'libxlang3_runtime.dylib'),'-Wl,-rpath,'+str(n),'-o',str(exe)]
with(w/'feasibility-build.log').open('w')as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
port=70000+os.getpid();nonce=uuid.uuid4().hex;ready=w/'feasibility-ready.json';assert not ready.exists()
paths=[w/'feasibility_server.py',w/'feasibility_client.cpp',Path(__file__).resolve(),exe,n/'xlang3',n/'libxlang3_runtime.dylib']
identities={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
(w/'feasibility-identities-before.json').write_text(json.dumps(dict(command=command,files=identities,scope='Native transport/clock correctness only; no Garnet inference or performance result'),indent=2)+'\n')
with(w/'feasibility-server.log').open('w')as out:
 server=subprocess.Popen([str(n/'xlang3'),'-B',str(w/'feasibility_server.py'),str(port),nonce,str(ready)],stdout=out,stderr=subprocess.STDOUT)
 try:
  for _ in range(150):
   assert server.poll()is None,'Native server exited before readiness'
   if ready.exists():break
   time.sleep(.1)
  else:raise RuntimeError('Native server readiness timeout')
  assert json.loads(ready.read_text())['pid']==server.pid
  with(w/'feasibility-client.log').open('w')as f:client=subprocess.run([str(exe),'lrpc:'+str(port),str(server.pid),nonce],stdout=f,stderr=subprocess.STDOUT,timeout=30)
  assert client.returncode==0,'Native client probe failed; retained logs'
  assert server.wait(timeout=10)==0
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in identities.items())
  (w/'feasibility-result.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_CPP_client_and_native_XLang3_server=True,pid_birth_verified=True,large_UTF8_preserved=True,shared_monotonic_epoch_10containment_checks=True,remote_exception_and_recovery=True,owned_server_clean_exit=True,identities_unchanged=True,no_model_or_GPU_inference_performance_claim=True),indent=2)+'\n')
  print('WS58 native LRPC/clock/owner/error/shutdown feasibility PASS; no model timing',flush=True)
 finally:
  if server.poll()is None:
   server.terminate()
   try:server.wait(timeout=5)
   except subprocess.TimeoutExpired:server.kill();server.wait()
