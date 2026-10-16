import os,sys,time,json,hashlib,pathlib,subprocess,shutil,runpy,traceback
w=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-prefix-cache');root=pathlib.Path('/Users/shawn/CantorAI');native=root/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor';plist=pathlib.Path('/Users/shawn/Library/LaunchAgents/com.cantorai.cantor.plist');api=runpy.run_path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-http.py')
labels=sys.argv[1:] or ['control-1','candidate-1','candidate-2','control-2'];records=[]
def command(args,check=True):return subprocess.run(args,check=check,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
def stop():
 command(['sh',str(native/'WorldSense/stop.sh')],False)
 command(['launchctl','bootout',job],False)
 for _ in range(150):
  if command(['launchctl','print',job],False).returncode:return
  time.sleep(.1)
 raise RuntimeError('Cantor bootout did not complete')
def start(label):
 command(['launchctl','bootstrap','gui/501',str(plist)])
 log=open(w/('server-'+label+'.log'),'ab')
 subprocess.Popen(['sh',str(native/'WorldSense/start.sh')],env=dict(os.environ,WORLDSENSE_ENABLE_BENCHMARK='1',WORLDSENSE_STORAGE_DIR=str(root/'out/data/worldsense')),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(60):
  try:return api['login']()
  except Exception:time.sleep(1)
 raise RuntimeError('WorldSense did not become ready')
def stage(source):
 temp=native/'libgarnet.dylib.benchmark-new';shutil.copy2(source,temp);os.replace(temp,native/'libgarnet.dylib')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

runtime_profiles=native/'WorldSense/Backend/Pipelines'
backup=w/'deployment-original-profiles';backup.mkdir(exist_ok=True)
for name in ['WorldSense-Skill-Inference.yml','WorldSense-Search-GPU.yml']:
 shutil.copy2(runtime_profiles/name,backup/name)
passed=False
try:
 stop()
 stage(root/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib')
 for name in ['WorldSense-Skill-Inference.yml','WorldSense-Search-GPU.yml']:
  shutil.copy2(root/'WorldSense/Backend/Pipelines/macos-metal'/name,runtime_profiles/name)
 token=start('deployment')
 with open(w/'deployment-client.log','w') as log:
  subprocess.run([str(native/'xlang3'),'-B',str(root/'WorldSense/tools/benchmark_recorded.py'),'http://localhost:9981',str(w/'deployment-cases.json'),str(w/'deployment.json'),'deployed-default-profile'],env=dict(os.environ,WORLDSENSE_BENCH_TOKEN=token),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
 data=json.loads((w/'deployment.json').read_text());assert data['valid'] and data['assignments_restored']
 golden=json.loads((w.parent/'worldsense-prefix-cache-confirm/paired-1.json').read_text())
 ref={}
 for item in golden['items']:
  for phase,call in item['result'].get('model_response',{}).items():
   if isinstance(call,dict) and call.get('status')=='ok':ref[(item['observation_id'],phase)]=call
 calls=[]
 for item in data['items']:
  for phase,call in item['result'].get('model_response',{}).items():
   if isinstance(call,dict)and call.get('status')=='ok':
    expected=ref[(item['observation_id'],phase)];assert expected['generated_token_ids']==call['generated_token_ids'] and expected['text']==call['text']
    config=call['serving_configuration'];assert config['prefill_chunk']==(64 if call['model_id']=='Qwen3.5-4B'else 16)
    assert config['text_prefix_cache']==(call['model_id']!='Qwen3.5-4B');calls.append(call)
 assert calls
 passed=True
 (w/'deployment-verification.json').write_text(json.dumps({'exact_default_profile_outputs':True,'profiles_enabled_without_benchmark_toggle':True,'library_sha256':sha(native/'libgarnet.dylib'),'returned_skill_calls':len(calls),'scope':'Deployment correctness smoke, not new matched speed evidence'},indent=2))
 print('DEPLOYED DEFAULT PROFILES VERIFIED',flush=True)
finally:
 stop()
 if not passed:
  stage(w/'before-libgarnet.dylib')
  for name in ['WorldSense-Skill-Inference.yml','WorldSense-Search-GPU.yml']:shutil.copy2(backup/name,runtime_profiles/name)
 command(['launchctl','bootstrap','gui/501',str(plist)])
 log=open(w/'deployment-normal-server.log','ab');env=dict(os.environ,WORLDSENSE_STORAGE_DIR=str(root/'out/data/worldsense'));env.pop('WORLDSENSE_ENABLE_BENCHMARK',None)
 subprocess.Popen(['sh',str(native/'WorldSense/start.sh')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('NORMAL SERVICE RESTARTED; candidate retained='+str(passed),flush=True)
