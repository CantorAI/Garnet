from pathlib import Path
import subprocess,hashlib,json
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-primary-simd-metadata/source');build=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-primary-simd-metadata-on')
command=['/usr/bin/c++','-std=c++17','-O3','-Wno-deprecated-declarations','-I',str(s/'src/backends/metal'),'-I',str(s/'src/model'),'-I',str(s/'third_party'),str(w/'simd_metadata_corpus_test.cpp'),str(s/'src/model/safetensors_index.cpp'),str(build/'libgarnet_metal.a'),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-o',str(build/'simd_metadata_corpus_test')]
with(w/'build-simd-corpus.log').open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(w/'simd-corpus-build-identity.json').write_text(json.dumps(dict(command=command,source_sha256=sha(w/'simd_metadata_corpus_test.cpp'),library_sha256=sha(build/'libgarnet_metal.a'),executable_sha256=sha(build/'simd_metadata_corpus_test')),indent=2)+'\n')
