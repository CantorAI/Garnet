"""Build corpus guards/fixtures and freeze complete learned-proof inputs."""
from pathlib import Path
import subprocess,sys,json,hashlib,datetime
w=Path(__file__).resolve().parent.parent/'worldsense-gpu-counter-profile';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-gpu-counter-profile/source';n=r/'out/build/macos-arm64-release/bin'
assert len(json.loads((w/'native-results.json').read_text()))==5 and all(v==0 for v in json.loads((w/'native-results.json').read_text()).values())
for name in ['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py']:
 with(w/('prepare-'+name+'.log')).open('w')as log:subprocess.run([sys.executable,str(w/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS '+name,flush=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
manifest=json.loads((w/'correctness-source-manifest.json').read_text());assert len(manifest)==1720 and all(sha(s/k)==v for k,v in manifest.items())
files={}
for model,assets in json.loads((w/'model-assets.json').read_text()).items():
 for rel,info in assets.items():
  p=r/'out/models'/model/rel;assert p.stat().st_size==info['bytes']and sha(p)==info['sha256'];files[str(p)]=info['sha256']
paths=[p for p in w.iterdir()if p.suffix in ['.py','.cpp','.inc']]
paths += [p for folder in ['fixture-1-8','fixture-2-64']for p in(w/folder).rglob('*')if p.is_file()]
paths += [w/'direct-upload-selection.json',n/'xlang3',n/'libxlang3_runtime.dylib',r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib',w.parent/'worldsense-primary-blocks/matrix-selection.json',w.parent/'worldsense-prefix-cache/state-guard.json',w.parent/'worldsense-prefix-cache/prefix-model-guard.json',w.parent/'worldsense-request-residency/vision-grid-off.json',w.parent/'worldsense-isolated-primary/api-state-guard-off.json']
paths += [w.parent/'skill-expansion'/name for name in ['car.jpg','traffic.jpg','bunny.jpg']]
for mode in ['off','primary','on','quiet','fault']:paths += [r/('out/build/garnet-metal-worldsense-gpu-counter-profile-'+mode)/name for name in ['libgarnet.dylib','garnet_gpu_pass_counter_test']]
paths += [r/'out/build/garnet-metal-worldsense-gpu-counter-profile-on'/name for name in ['primary_corpus_test','direct_float_corpus_test','libgarnet_metal.a']]
files.update({str(p):sha(p)for p in paths})
(w/'learned-identities-before.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=manifest,external=files,scope='All learned assets, source/runtime/libraries/drivers/oracles and corpus executables; full timing host closure remains before actual capture.'),indent=2)+'\n')
print('WS62 full learned inputs frozen; no inference dispatched',flush=True)
