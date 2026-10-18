from pathlib import Path
import subprocess,json,hashlib,shutil,os,time,runpy,traceback
w=Path(__file__).resolve().parent; task=w.parent.parent;root=Path('/Users/shawn/CantorAI');repo=root/'Garnet';build=root/'out/build/garnet-main-macos-verify-20261018';native=root/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor'
results=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def run(label,args):
 print('START',label,flush=True);started=time.monotonic()
 with (w/(label+'.log')).open('w')as f:p=subprocess.run([str(a)for a in args],stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(label=label,exit=p.returncode,seconds=time.monotonic()-started));(w/'validation-progress.json').write_text(json.dumps(results,indent=2));print('FINISH',label,p.returncode,flush=True);assert p.returncode==0,label

def stop():
 subprocess.run(['sh',str(native/'WorldSense/stop.sh')],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
 subprocess.run(['launchctl','bootout',job],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
 for _ in range(150):
  if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:return
  time.sleep(.1)
 raise RuntimeError('Cantor did not unload')

shutil.copy2(task/'work/worldsense-prefix-cache/prefix-safety-guard.json',w/'frozen-ws10-prefix-safety-guard.json')
shutil.copy2(task/'work/worldsense-prefix-cache/deployment-cases.json',w/'worldsense-cases.json')
original=json.loads((w/'inputs-before-validation.json').read_text());files=dict(original['files'])
for p,s in files.items():assert sha(p)==s,p
for manifest in ['image-base-cases.json','image-edge-cases.json']:
 for c in json.loads((w/manifest).read_text()):
  for key in ['image','reference']:files[c[key]]=sha(c[key])
assets=json.loads((task/'work/worldsense-owned-storage-local/model-assets.json').read_text())
for model,entries in assets.items():
 for name,info in entries.items():
  p=root/'out/models'/model/name;actual=sha(p);assert actual==info['sha256'],str(p);files[str(p)]=actual
for p in [Path(__file__),repo/'test2026/metal/test_native_image.py',native/'xlang3',native/'libxlang3_runtime.dylib',w/'worldsense-cases.json',w/'frozen-ws10-prefix-safety-guard.json',*build.glob('test_*')]:
 if p.is_file():files[str(p)]=sha(p)
def flags(p):
 return dict(line.strip().split(':BOOL=',1)for line in p.read_text().splitlines()if line.startswith('GARNET_')and ':BOOL='in line)
a=flags(root/'out/build/garnet-metal-worldsense-prefix-cache/CMakeCache.txt');b=flags(build/'CMakeCache.txt');assert all(b.get(k)==v for k,v in a.items()),'WS10 flags differ';assert all(v=='OFF'for k,v in b.items()if k not in a),'Extra experimental flag enabled'
frozen=dict(original,main_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),metal_commit=subprocess.check_output(['git','rev-parse','origin/feature/qwen35-metal'],cwd=repo,text=True).strip(),files=files,flags=b,model_assets=assets)
(w/'inputs-frozen.json').write_text(json.dumps(frozen,indent=2));print('FROZEN',len(files),'input identities; WS10 profile matched',flush=True)
try:
 stop()
 run('ctest',['/opt/homebrew/bin/ctest','--test-dir',build,'--output-on-failure'])
 for name in ['base','edge']:
  run('image-'+name,[native/'xlang3','-B',repo/'test2026/metal/test_native_image.py',build,root/'out/models/Qwen3.5-4B',w/('image-'+name+'-cases.json'),w/('image-'+name+'-results.json')])
 for label in ['state','prefix-model','prefix-safety']:
  run(label,[native/'xlang3','-B',w/(label.replace('-','_')+'_guard.py')])
 checks={}
 for label in ['state','prefix-model']:
  new=json.loads((w/(label+'-guard.json')).read_text());old=json.loads((w/('frozen-ws10-'+label+'-guard.json')).read_text())
  assert len(new['records'])==len(old['records'])
  for x,y in zip(new['records'],old['records']):
   for k in ['states','generated_token_ids','prompt_sha256','policy','prompt_index','actual_max_prefill_chunk']:
    if k in y:assert x[k]==y[k],label+':'+k
  checks[label+'_all_layers_tokens_match_frozen_WS10']=True
 new=json.loads((w/'prefix-safety-guard.json').read_text());old=json.loads((w/'frozen-ws10-prefix-safety-guard.json').read_text());assert new==old;checks['prefix_safety_matches_frozen_WS10']=True
 # Full native projection output, not rounded scalar summaries.
 prior=root/'out/build/garnet-metal-worldsense-minimal-selective-defaults-off/primary_projection_words.bin';fresh=build/'primary_projection_words.bin';assert prior.read_bytes()==fresh.read_bytes();checks['projection_full_stream_bitwise_equal']=dict(bytes=fresh.stat().st_size,sha256=sha(fresh))
 (w/'correctness-verification.json').write_text(json.dumps(checks,indent=2))
 # Exercise actual WorldSense -> Galaxy -> Garnet on the newly built main library.
 temporary=native/'libgarnet.dylib.main-verify-new';shutil.copy2(build/'libgarnet.dylib',temporary);os.replace(temporary,native/'libgarnet.dylib')
 subprocess.run(['launchctl','bootstrap','gui/501','/Users/shawn/Library/LaunchAgents/com.cantorai.cantor.plist'],check=True)
 run('wait-main',[native/'xlang3','-B',w/'wait_cantor.py'])
 log=(w/'worldsense-server.log').open('ab');subprocess.Popen(['sh',str(native/'WorldSense/start.sh')],env=dict(os.environ,WORLDSENSE_ENABLE_BENCHMARK='1',WORLDSENSE_STORAGE_DIR=str(root/'out/data/worldsense')),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 api=runpy.run_path(str(task/'work/worldsense-http.py'))
 for _ in range(60):
  try:token=api['login']();break
  except Exception:time.sleep(1)
 else:raise RuntimeError('WorldSense not ready')
 env=dict(os.environ,WORLDSENSE_BENCH_TOKEN=token)
 print('START WorldSense main fixed-frame suite',flush=True)
 with(w/'worldsense-client.log').open('w')as f:p=subprocess.run([str(native/'xlang3'),'-B',str(root/'WorldSense/tools/benchmark_recorded.py'),'http://localhost:9981',str(w/'worldsense-cases.json'),str(w/'worldsense-main.json'),'main-macos-merge-verification'],env=env,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,'WorldSense benchmark failed';print('FINISH WorldSense main fixed-frame suite',flush=True)
except Exception:
 traceback.print_exc();raise
finally:
 stop();temporary=native/'libgarnet.dylib.main-restore';shutil.copy2(w/'before-libgarnet.dylib',temporary);os.replace(temporary,native/'libgarnet.dylib')
 run('restore-normal',['/usr/bin/python3',w/'restore_normal.py'])
# Freeze inputs again only after all inference and timings have ended.
assert all(sha(p)==s for p,s in files.items()),'Consumed inputs changed'
(w/'inputs-unchanged.json').write_text(json.dumps(dict(passed=True,identities=len(files)),indent=2));print('ALL MAIN VALIDATION JOBS TERMINAL; inputs unchanged; normal service restored',flush=True)
