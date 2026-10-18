"""Actual-model proof input freeze before the first WS61 model execution."""
from pathlib import Path
import json,hashlib,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-validated-text-worker/source';old=r/'out/research/worldsense-prefix-cache/source';n=r/'out/build/macos-arm64-release/bin'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
parent=json.loads((w/'gateway-source-manifest.json').read_text());legacy=json.loads((w/'validated-source-manifest.json').read_text())
assert all(sha(s/k)==v for k,v in parent.items())and all(sha(old/k)==v for k,v in legacy.items())
external={}
for model,files in json.loads((w/'model-assets.json').read_text()).items():
 for rel,info in files.items():
  p=r/'out/models'/model/rel;assert p.stat().st_size==info['bytes']and sha(p)==info['sha256'];external[str(p)]=info['sha256']
paths=[n/'xlang3',n/'libxlang3_runtime.dylib',r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib',w.parent/'worldsense-prefix-cache/prefix-model-guard.json',w.parent/'worldsense-isolated-primary/api-state-guard-off.json']
paths += [p for p in w.glob('*.py')if p.is_file()]
for role in ['off','on','primary','fault']:
 b=r/('out/build/garnet-metal-worldsense-validated-text-worker-'+role);paths.append(b/'libgarnet.dylib')
 if role!='off':paths.append(b/'libgarnet_worker_host.dylib')
external.update({str(p):sha(p)for p in paths})
(w/'first-model-identities-before.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_source=parent,legacy_source=legacy,external=external,scope='Before first actual model calls; no timing or complete build-host closure claim. All captured files frozen through the text controller.'),indent=2)+'\n')
print('WS61 parent/old source, four-role libraries/nativehosts, runtime, learned assets, oracles and drivers frozen before first model calls',flush=True)
