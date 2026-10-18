from pathlib import Path
import subprocess,json,hashlib,os,time,secrets,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-thread-qos';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-thread-qos/source';n=r/'out/build/macos-arm64-release/bin';cpp=Path(__file__).with_name('ws59_inherited_client.cpp');exe=w/'inherited-client';command=['/usr/bin/c++','-std=c++17','-O2','-I',str(r/'xlang3/sdk'),'-I',str(s/'third_party'),str(cpp),str(n/'libxlang3_runtime.dylib'),'-Wl,-rpath,'+str(n),'-o',str(exe)]
with(w/'inherited-build.log').open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
(w/'inherited-driver-identities.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [cpp,Path(__file__).resolve(),exe,s/'runtime/native_text_worker.py',r/'out/build/garnet-metal-worldsense-thread-qos-off/libgarnet.dylib',n/'xlang3',n/'libxlang3_runtime.dylib']}),indent=2)+'\n')
ready=w/'inherited-ready.json';assert not ready.exists();birth=secrets.token_hex(32);port=(1<<46)+os.getpid()*(1<<20);env=dict(os.environ,GARNET_NATIVE_WORKER_PARENT=str(os.getpid()))
with(w/'inherited-worker.log').open('w')as log:
 proc=subprocess.Popen([str(n/'xlang3'),'-B',str(s/'runtime/native_text_worker.py'),str(port),str(os.getpid()),birth,str(r/'out/build/garnet-metal-worldsense-thread-qos-off'),str(s),str(ready),'0'],stdout=log,stderr=subprocess.STDOUT,env=env)
 try:
  deadline=time.monotonic()+20
  while not ready.exists():assert proc.poll()is None and time.monotonic()<deadline;time.sleep(.02)
  info=json.loads(ready.read_text());assert info['support']['pid']==proc.pid and info['support']['parent_pid']==os.getpid()and info['birth_id']==birth
  with(w/'inherited-client.log').open('w')as out:subprocess.run([str(exe),'lrpc:'+str(port),str(proc.pid),birth,str(w/'inherited-mode-result.json')],stdout=out,stderr=subprocess.STDOUT,timeout=30,check=True)
  assert proc.wait(timeout=10)==0
 finally:
  if proc.poll()is None:proc.terminate();proc.wait(timeout=10)
print('Native inherited runtime-mode, zero scopes, structured model-not-ready error, no retry and clean exit PASS; no model/GPU inference')
