"""Complete before/after identity-frozen numerical proof; no timing claim."""
from pathlib import Path
import subprocess,sys,json,hashlib,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI')
s=r/'out/research/worldsense-dense-prefetch/source';n=r/'out/build/macos-arm64-release/bin'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
m=json.loads((w/'correctness-source-manifest.json').read_text())
assert len(m)==1720 and all(sha(s/k)==v for k,v in m.items())
assets=json.loads((w/'model-assets.json').read_text());external={}
for model,files in assets.items():
 for name,record in files.items():
  p=r/'out/models'/model/name
  assert p.stat().st_size==record['bytes']and sha(p)==record['sha256'],str(p)
  external[str(p)]=record['sha256']
paths=[n/'xlang3',n/'libxlang3_runtime.dylib']
paths += [w.parent/'skill-expansion'/name for name in ['car.jpg','traffic.jpg','bunny.jpg']]
paths += [w.parent/'worldsense-prefix-cache'/name for name in ['state-guard.json','prefix-model-guard.json']]
paths += [w.parent/'worldsense-request-residency/vision-grid-off.json',w.parent/'worldsense-primary-blocks/matrix-selection.json']
paths += [p for folder in ['fixture-1-8','fixture-2-64']for p in (w/folder).rglob('*')if p.is_file()]
paths += [r/'out/build/garnet-metal-worldsense-dense-prefetch-on'/name for name in ['primary_corpus_test','direct_float_corpus_test']]
external.update({str(p.resolve()):sha(p)for p in paths})
identity=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=m,drivers={str(p):sha(p)for p in w.glob('*.py')},libraries={role:sha(r/('out/build/garnet-metal-worldsense-dense-prefetch-'+role)/'libgarnet.dylib')for role in ['off','primary','on','fault']},external_inputs=external,scope='Numerical proof inputs and native model/runtime/driver/source/library identities, not complete build-host closure or a performance measurement. Full host/toolchain dependency freeze remains required before timing.')
(w/'full-proof-identities-before.json').write_text(json.dumps(identity,indent=2)+'\n')
print('LEARNED_MODELS_INPUTS_SOURCE_DRIVERS_LIBRARIES_IDENTITY_PASS',flush=True)
for script in ['native_controller.py','full_correctness.py']:
 print('START '+script,flush=True)
 with(w/('proof-'+script.replace('.py','.log'))).open('w')as log:
  subprocess.run([sys.executable,str(w/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS '+script,flush=True)
assert all(sha(s/k)==v for k,v in m.items())
assert all(sha(Path(p))==v for p,v in identity['drivers'].items())
assert all(sha(Path(p))==v for p,v in external.items())
assert all(sha(r/('out/build/garnet-metal-worldsense-dense-prefetch-'+role)/'libgarnet.dylib')==v for role,v in identity['libraries'].items())
(w/'full-proof-complete.json').write_text(json.dumps(dict(all_original_native_model_vision_session_corpus_guards_pass=True,new_raw_dense_prefetch_full_word_state_and_actual_route_guards_pass=True,all_captured_source_driver_library_model_runtime_inputs_unchanged=True,no_speed_result=True),indent=2)+'\n')
print('Full WS60 model/corpus/vision/session/rawprefetch correctness PASS; no speed claim',flush=True)
