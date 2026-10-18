from pathlib import Path
import subprocess,json,hashlib
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-packed-request-residency/source';b=r/'out/build/garnet-metal-worldsense-packed-request-residency-on'
cmd=['/usr/bin/c++','-std=c++17','-O3','-Wno-deprecated-declarations','-I',str(s/'src/backends/metal'),'-I',str(s/'src/model'),'-I',str(s/'third_party'),str(w/'packed_lease_corpus_test.cpp'),str(s/'src/model/safetensors_index.cpp'),str(b/'libgarnet_metal.a'),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-o',str(b/'packed_lease_corpus_test')]
with(w/'build-packed-lease-corpus.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
(w/'packed-lease-corpus-build-identity.json').write_text(json.dumps(dict(command=cmd,source_sha256=hashlib.sha256((w/'packed_lease_corpus_test.cpp').read_bytes()).hexdigest(),library_sha256=hashlib.sha256((b/'libgarnet_metal.a').read_bytes()).hexdigest(),executable_sha256=hashlib.sha256((b/'packed_lease_corpus_test').read_bytes()).hexdigest()),indent=2)+'\n')
print('Packed lease learned corpus executable built; not executed')
