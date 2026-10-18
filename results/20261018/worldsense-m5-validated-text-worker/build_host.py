"""Compile native-only instrumentation package; never invokes inference."""
from pathlib import Path
import subprocess,hashlib,json
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');p=w/'native-host';n=r/'out/build/macos-arm64-release/bin';s=r/'out/research/worldsense-prefix-cache/source'
cmd=['/usr/bin/c++','-std=c++17','-O2','-fobjc-arc','-fno-fast-math','-ffp-contract=off','-Wno-deprecated-declarations','-dynamiclib','-I',str(r/'xlang3/sdk'),'-I',str(s/'third_party'),str(p/'worker_host.mm'),str(p/'native_worker_support.cpp'),'-DWORKER_HOST_ROOT="'+str(p)+'"','-DWORKER_LEGACY_LIBRARY="'+str(r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib')+'"','-DWORKER_LEGACY_SOURCE="'+str(s)+'"',str(n/'libxlang3_runtime.dylib'),'-Wl,-rpath,'+str(n),'-framework','Foundation','-framework','Metal','-o',str(p/'libgarnet_worker_host.dylib')]
with(w/'build-host.log').open('w')as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
(w/'native-host-build-identity.json').write_text(json.dumps(dict(command=cmd,sources={str(q):sha(q)for q in p.iterdir()if q.suffix in ['.mm','.cpp','.h']},library_sha256=sha(p/'libgarnet_worker_host.dylib'),no_Garnet_binary_or_source_change=True,no_model_or_GPU_inference=True),indent=2)+'\n')
print('Native instrumentation sidecar built; no model or speed result',flush=True)
