"""Compile proof executables and freeze explicit inputs before first learned execution."""
from pathlib import Path
import subprocess,sys,json,hashlib,datetime,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-packed-request-residency';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-packed-request-residency/source';n=r/'out/build/macos-arm64-release/bin';load=lambda p:json.loads(p.read_text())
assert load(w/'native-independent-audit.json')['native_tests_pass']==54
assert not(w/'learned-identities-before.json').exists(),'Freeze exactly once before first learned dispatch'
for p in w.glob('*.py'):ast.parse(p.read_text())
ast.parse((b/'work/worldsense-opt/verify_learned_ws69.py').read_text())
for name in ['build_corpus.py','build_packed_lease_corpus.py','build_direct_corpus.py','make_resident_fixtures.py']:
 with(w/('prepare-'+name+'.log')).open('w')as log:subprocess.run([sys.executable,str(w/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS '+name,flush=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
source=load(w/'correctness-source-manifest.json');assert len(source)==1721 and all(sha(s/k)==v for k,v in source.items())
inputs={}
for model,assets in load(w/'model-assets.json').items():
 for name,info in assets.items():
  p=r/'out/models'/model/name;assert p.stat().st_size==info['bytes']and sha(p)==info['sha256'];inputs[str(p)]=info['sha256']
names=['build_corpus.py','build_packed_lease_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','primary_corpus_test.cpp','packed_lease_corpus_test.cpp','direct_float_corpus_test.cpp','request_session_guard.py','request_full_session_guard.py','packed_layer_session_guard.py','packed_full_session_guard.py','full_correctness.py','restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib','direct-upload-selection.json','model-assets.json','learned-guard-adaptation-proof.json','native-independent-audit.json','native-design-before-implementation.txt','protocol.json','correctness-source-manifest.json','source-transformations.json','source-inverse-proof.json','packed-context-adaptation-proof.json','packed-session-adaptation-proof.json']
files=[w/name for name in names]
files += [w.parent/'worldsense-request-weight-residency'/name for name in load(w/'learned-guard-adaptation-proof.json')]
files += [p for folder in ['fixture-1-8','fixture-2-64']for p in(w/folder).rglob('*')if p.is_file()]
files += [b/'work/worldsense-opt'/name for name in ['prepare_full_ws69.py','prepare_learned_ws69.py','verify_learned_ws69.py','verify_native_ws69.py']]
files += [n/'xlang3',n/'libxlang3_runtime.dylib',r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib',w.parent/'worldsense-primary-blocks/matrix-selection.json',w.parent/'worldsense-prefix-cache/state-guard.json',w.parent/'worldsense-prefix-cache/prefix-model-guard.json',w.parent/'worldsense-request-residency/vision-grid-off.json']
files += [w.parent/'skill-expansion'/name for name in ['car.jpg','traffic.jpg','bunny.jpg']]
for role in ['off','primary','on','fault','compat56','compat56-fault']:
 build=r/('out/build/garnet-metal-worldsense-packed-request-residency-'+role);files += [p for p in build.iterdir()if p.is_file()and(p.name.startswith('garnet_')or p.name in ['libgarnet.dylib','libgarnet_metal.a','CMakeCache.txt'])]
for role,names in [('compat56',['primary_corpus_test','direct_float_corpus_test']),('on',['packed_lease_corpus_test'])]:files += [r/('out/build/garnet-metal-worldsense-packed-request-residency-'+role)/name for name in names]
inputs.update({str(p):sha(p)for p in files})
assert not any(Path(k).name in ['native-results.json','resident-results.json','api-correctness-result.json','worldsense-http.py']for k in inputs)
(w/'learned-identities-before.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=source,external=inputs,scope='Explicit frozen inputs only, including outer verifier/controller, full model assets/oracles and all six binaries. Generated results excluded; no inference yet.'),indent=2)+'\n')
print('WS69 full learned input closure frozen; no learned execution yet',flush=True)
