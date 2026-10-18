"""Original native math tests plus counter metadata/transaction tests, no speed."""
from pathlib import Path
import subprocess,json,hashlib,time,sys,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-gpu-counter-profile/source';n=r/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor';roles=['off','primary','on','quiet','fault']
manifest=json.loads((w/'correctness-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
paths=[Path(__file__).resolve(),w/'build.py',n/'xlang3',n/'libxlang3_runtime.dylib']
for mode in roles:
 b=r/('out/build/garnet-metal-worldsense-gpu-counter-profile-'+mode);paths += [b/'libgarnet.dylib',b/'garnet_gpu_pass_counter_test']
identities={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
(w/'native-identities-before.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=identities),indent=2)+'\n')
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Normal service did not unload')
try:
 results={};words={};programs={}
 for mode in roles:
  b=r/('out/build/garnet-metal-worldsense-gpu-counter-profile-'+mode)
  tests=json.loads(subprocess.check_output(['/opt/homebrew/bin/ctest','--test-dir',str(b),'--show-only=json-v1'],text=True))['tests']
  assert len(tests)==8 and sum(t['name']=='metal_gpu_pass_counter_reference'for t in tests)==1
  with(w/('native-'+mode+'.log')).open('w')as log:status=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(b),'--output-on-failure'],stdout=log,stderr=subprocess.STDOUT,timeout=600)
  results[mode]=status.returncode;(w/'native-results.json').write_text(json.dumps(results,indent=2)+'\n');print('Native8 tests '+mode+' exit '+str(status.returncode),flush=True);assert status.returncode==0
  words[mode]=(b/'primary_projection_words.bin').read_bytes();programs[mode]={};prior_mode='on'if mode=='quiet'else mode
  for kind in ['main','prefill']:
   data=(b/(kind+'_shader_program.metal')).read_bytes();assert data==(w.parent/'worldsense-isolated-primary'/('actual-'+prior_mode+'-'+kind+'-program.metal')).read_bytes()
   (w/('actual-'+mode+'-'+kind+'-program.metal')).write_bytes(data);programs[mode][kind]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),exact_WS54=True)
  trace=b/'gpu-pass-counter-test.jsonl'
  if mode=='quiet':assert not trace.exists()
  else:
   assert trace.is_file();(w/('native-counter-'+mode+'.jsonl')).write_bytes(trace.read_bytes())
 assert all(value==words['off']for value in words.values())
 assert words['off']==(r/'out/build/garnet-metal-worldsense-isolated-primary-off/primary_projection_words.bin').read_bytes()
 (w/'native-word-comparison.json').write_text(json.dumps(dict(records=186,all_five_builds_FP32_words_identical=True,bytes=len(words['off']),sha256=hashlib.sha256(words['off']).hexdigest()),indent=2)+'\n')
 (w/'actual-program-proof.json').write_text(json.dumps(dict(all_five_actual_MAIN_and_prefill_sources_exact_seed=True,programs=programs,scope='Actual constructor MSL source, not GPU ISA/performance identity.'),indent=2)+'\n')
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in identities.items())
 assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
 print('Five roles original native math,186 full word records,actual shaders and GPU-counter transactions PASS',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
