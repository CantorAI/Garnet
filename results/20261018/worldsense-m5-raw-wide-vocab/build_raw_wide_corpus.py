from pathlib import Path
import subprocess,hashlib,json
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-raw-wide-vocab/source');b=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-raw-wide-vocab-on')
assert 'GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB:BOOL=ON'in(b/'CMakeCache.txt').read_text()
cmd=['/usr/bin/c++','-std=c++17','-O3','-DGARNET_METAL_PRIMARY_RAW_WIDE_VOCAB=1','-Wno-deprecated-declarations','-I',str(s/'src/backends/metal'),'-I',str(s/'src/model'),'-I',str(s/'third_party'),str(w/'raw_wide_corpus_test.cpp'),str(s/'src/model/safetensors_index.cpp'),str(b/'libgarnet_metal.a'),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-o',str(b/'raw_wide_corpus_test')]
with(w/'build-raw-wide-corpus.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
(w/'raw-wide-corpus-build-identity.json').write_text(json.dumps({'command':cmd,'source_sha256':hashlib.sha256((w/'raw_wide_corpus_test.cpp').read_bytes()).hexdigest(),'library_sha256':hashlib.sha256((b/'libgarnet_metal.a').read_bytes()).hexdigest(),'executable_sha256':hashlib.sha256((b/'raw_wide_corpus_test').read_bytes()).hexdigest()},indent=2));print('GPU corpus correctness executable built',flush=True)
