"""Full original learned guards plus BOTH new helpers, never speed evidence."""
from pathlib import Path
import subprocess,sys,json,hashlib,time
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';s=r/'out/research/worldsense-primary-simd-metadata/source';load=lambda p:json.loads(p.read_text());manifest=load(w/'correctness-source-manifest.json')
assert load(w/'native-independent-audit.json')['native_tests_pass']==50
assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
inputs=load(w/'learned-identities-before.json')['external']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
assert all(sha(Path(k))==v for k,v in inputs.items())
def run(command,name):
 with(w/name).open('w')as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
job='gui/501/com.cantorai.cantor'
try:
 subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
 for _ in range(200):
  if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
  time.sleep(.1)
 else:raise RuntimeError('Service did not unload')
 compat=r/'out/build/garnet-metal-worldsense-primary-simd-metadata-compat67';candidate=r/'out/build/garnet-metal-worldsense-primary-simd-metadata-on';selection=w.parent/'worldsense-primary-blocks/matrix-selection.json'
 for executable,build,request,output in [('primary_corpus_test',compat,selection,'corpus-GPU-bits'),('simd_metadata_corpus_test',candidate,selection,'simd-metadata-corpus'),('direct_float_corpus_test',compat,w/'direct-upload-selection.json','direct-upload-corpus'),('dual_pair_corpus_test',compat,selection,'dual-pair-corpus')]:
  run([str(build/executable),str(request),str(w/(output+'.json'))],output+'.log');print('PASS '+output,flush=True)
 run([sys.executable,str(w/'resident_controller.py')],'trained-resident-controller.log');print('PASS full resident/private/shared/rebuild matrix',flush=True)
 for mode in ['off','on','fault','compat67']:run([str(n/'xlang3'),'-B',str(w/'linear_guard.py'),mode],'trained-linear-'+mode+'.log')
 print('PASS original trained linear guards',flush=True)
 for mode in ['off','on','compat67']:run([str(n/'xlang3'),'-B',str(w/'vision_grid_guard.py'),mode],'vision-grid-'+mode+'.log')
 print('PASS original vision grids',flush=True)
 for mode in ['off','primary','on','compat67']:
  run([str(n/'xlang3'),'-B',str(w/'api_state_guard.py'),mode],'api-state-guard-'+mode+'.log');print('PASS full state/token native API '+mode,flush=True)
 assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in manifest.items())and all(sha(Path(k))==v for k,v in inputs.items())
 (w/'api-correctness-result.json').write_text(json.dumps(dict(original32VLM28text_states_tokens_vision_prefix_capacity_and_isolation_pass=True,source1726_and_all_enumerated_inputs_unchanged=True,all_four_actual_API_modes_complete=True,no_speed_claim=True),indent=2)+'\n');print('WS68 full learned controller PASS',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
