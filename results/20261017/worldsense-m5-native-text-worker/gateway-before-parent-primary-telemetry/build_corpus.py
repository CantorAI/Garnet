from pathlib import Path
import subprocess,hashlib,json
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-native-text-worker/source');b=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-native-text-worker-on')
cmd=['/usr/bin/c++','-std=c++17','-O3','-Wno-deprecated-declarations','-I',str(s/'src/backends/metal'),'-I',str(s/'src/model'),'-I',str(s/'third_party'),str(w/'primary_corpus_test.cpp'),str(s/'src/model/safetensors_index.cpp'),str(b/'libgarnet_metal.a'),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-o',str(b/'primary_corpus_test')]
with(w/'build-corpus.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
(w/'corpus-build-identity.json').write_text(json.dumps({'command':cmd,'source_sha256':hashlib.sha256((w/'primary_corpus_test.cpp').read_bytes()).hexdigest(),'library_sha256':hashlib.sha256((b/'libgarnet_metal.a').read_bytes()).hexdigest(),'executable_sha256':hashlib.sha256((b/'primary_corpus_test').read_bytes()).hexdigest()},indent=2));print('GPU corpus correctness executable built',flush=True)
