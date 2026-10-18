"""Build then execute native QoS API feasibility; no Metal/model measurement."""
from pathlib import Path
import subprocess,json,hashlib,os,time,secrets,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';source=r/'out/research/worldsense-native-text-worker/source';commands=[]
base=['/usr/bin/c++','-std=c++17','-O2','-I',str(r/'xlang3/sdk'),'-I',str(source/'third_party')]
for command in [base+['-dynamiclib',str(w/'qos_probe.cpp'),str(n/'libxlang3_runtime.dylib'),'-Wl,-rpath,'+str(n),'-o',str(w/'libqos_probe.dylib')],base+[str(w/'qos_client.cpp'),str(n/'libxlang3_runtime.dylib'),'-Wl,-rpath,'+str(n),'-o',str(w/'qos-client')]]:
 with(w/('build-'+str(len(commands))+'.log')).open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 commands.append(command)
paths=[w/'qos_probe.cpp',w/'qos_client.cpp',w/'qos_server.py',Path(__file__).resolve(),w/'libqos_probe.dylib',w/'qos-client',n/'xlang3',n/'libxlang3_runtime.dylib'];(w/'feasibility-identities.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands=commands,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}),indent=2)+'\n')
port=(1<<47)+os.getpid()*(1<<20)+1;ready=w/'qos-ready.json';assert not ready.exists();birth=secrets.token_hex(32)
with(w/'native-server.log').open('w')as log:
 proc=subprocess.Popen([str(n/'xlang3'),'-B',str(w/'qos_server.py'),str(port),str(ready),birth],stdout=log,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+20
  while not ready.exists():assert proc.poll()is None and time.monotonic()<deadline;time.sleep(.02)
  identity=json.loads(ready.read_text());assert identity['pid']==proc.pid and identity['parent_pid']==os.getpid()and identity['birth']==birth
  with(w/'native-client.log').open('w')as out:subprocess.run([str(w/'qos-client'),'lrpc:'+str(port),str(proc.pid),birth,str(w/'native-rpc-qos.json')],stdout=out,stderr=subprocess.STDOUT,check=True,timeout=30)
  assert proc.wait(timeout=10)==0
 finally:
  if proc.poll()is None:proc.terminate();proc.wait(timeout=10)
main=json.loads((w/'native-main-requested-qos.json').read_text());rpc=json.loads((w/'native-rpc-qos.json').read_text());assert main['scope_probe']['native_API_scope_contract_pass']and rpc['rpc_scope_probe']['native_API_scope_contract_pass']
(w/'feasibility-result.json').write_text(json.dumps(dict(native_main_scope_and_RPC_scope_contract_pass=True,main_requested=main['context'],rpc_requested=rpc['rpc_requested_context'],owned_server_exit=proc.returncode,no_Garnet_or_engine_source_change=True,no_Metal_or_model_inference=True,no_speed_claim=True),indent=2)+'\n');print('Native requested-QoS and RAII nested/idempotent/unwind API feasibility PASS; no GPU/model/performance claim',flush=True)
