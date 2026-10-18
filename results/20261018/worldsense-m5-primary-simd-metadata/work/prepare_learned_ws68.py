"""Build proof tools/fixtures, then explicitly freeze INPUTS only before any learned call."""
from pathlib import Path
import subprocess,sys,hashlib,json,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-primary-simd-metadata';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-simd-metadata/source';n=r/'out/build/macos-arm64-release/bin';load=lambda p:json.loads(p.read_text())
assert load(w/'native-independent-audit.json')['native_tests_pass']==50
for name in ['build_corpus.py','build_simd_corpus.py','build_direct_corpus.py','build_dual_pair_corpus.py','make_resident_fixtures.py']:
 with(w/('prepare-'+name+'.log')).open('w')as log:subprocess.run([sys.executable,str(w/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS '+name,flush=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
source=load(w/'correctness-source-manifest.json');assert len(source)==1726 and all(sha(s/k)==v for k,v in source.items())
inputs={}
for model,assets in load(w/'model-assets.json').items():
 for name,info in assets.items():
  p=r/'out/models'/model/name;assert p.stat().st_size==info['bytes']and sha(p)==info['sha256'];inputs[str(p)]=info['sha256']
files=[w/name for name in ['build_corpus.py','build_simd_corpus.py','build_direct_corpus.py','build_dual_pair_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','primary_corpus_test.cpp','direct_float_corpus_test.cpp','dual_pair_corpus_test.cpp','simd_metadata_corpus_test.cpp','full_correctness.py','restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib','direct-upload-selection.json','model-assets.json','learned-guard-adaptation-proof.json','native-independent-audit.json','native-design-before-implementation.txt','protocol.json','correctness-source-manifest.json','source-transformations.json','simd-shader-inverse-proof.json']]
files += [p for folder in ['fixture-1-8','fixture-2-64']for p in(w/folder).rglob('*')if p.is_file()]
files += [b/'work/worldsense-opt'/name for name in ['prepare_full_ws68.py','prepare_learned_ws68.py','verify_learned_ws68.py','verify_native_ws68.py']]
files += [n/'xlang3',n/'libxlang3_runtime.dylib',r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib',w.parent/'worldsense-primary-blocks/matrix-selection.json',w.parent/'worldsense-prefix-cache/state-guard.json',w.parent/'worldsense-prefix-cache/prefix-model-guard.json',w.parent/'worldsense-request-residency/vision-grid-off.json']
files += [w.parent/'skill-expansion'/name for name in ['car.jpg','traffic.jpg','bunny.jpg']]
for role in ['off','primary','on','fault','compat67']:
 build=r/('out/build/garnet-metal-worldsense-primary-simd-metadata-'+role);files += [p for p in build.iterdir()if p.is_file()and(p.name.startswith('garnet_')or p.name in ['libgarnet.dylib','libgarnet_metal.a','CMakeCache.txt'])]
for role,names in [('compat67',['primary_corpus_test','direct_float_corpus_test','dual_pair_corpus_test']),('on',['simd_metadata_corpus_test'])]:
 files += [r/('out/build/garnet-metal-worldsense-primary-simd-metadata-'+role)/name for name in names]
inputs.update({str(p):sha(p)for p in files})
assert str(w/'native-results.json')not in inputs and str(w/'resident-results.json')not in inputs
(w/'learned-identities-before.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=source,external=inputs,scope='Explicit frozen inputs and outer verifier/controller before first learned call; generated results deliberately excluded. Original corpus COMPAT67, both new full201helpers ON. No inference yet.'),indent=2)+'\n')
print('WS68 complete learned proof input closure frozen before first learned execution',flush=True)
