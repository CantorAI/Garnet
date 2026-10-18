"""Sequential RAW-first build. Never start timing or inference tests here."""
from pathlib import Path
import subprocess,json,hashlib,sys,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-native-text-worker/source';raw=r/'out/build/garnet-metal-worldsense-native-text-worker-off';n=r/'out/build/macos-arm64-release/bin'
flags=['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_NATIVE_TEXT_WORKER']
m={str(p.relative_to(s)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(s.rglob('*'))if p.is_file()and '__pycache__'not in p.parts};(w/'gateway-source-manifest.json').write_text(json.dumps(m,indent=2)+'\n');(w/'gateway-build-driver-identity.json').write_text(json.dumps(dict(driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_files=len(m),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
for mode in sys.argv[1:]or['off','on','primary','fault']:
 assert mode in ['off','on','primary','fault'];b=r/('out/build/garnet-metal-worldsense-native-text-worker-'+mode);primary=mode!='off';vector=mode in ['on','fault'];values={k:False for k in flags}
 for k in flags[:3]:values[k]=primary
 for k in flags[3:9]:values[k]=vector
 values[flags[9]]=primary
 command=['/opt/homebrew/bin/cmake','-S',str(s),'-B',str(b),'-C',str(w/'build-profile/MetalWorldSense4BPrefix.cmake'),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DXLANG3_SDK_ROOT='+str(r/'xlang3/sdk'),'-DGARNET_XLANG3_RUNTIME_LIBRARY='+str(n/'libxlang3_runtime.dylib'),'-DGARNET_METAL_TEXT_WORKER_RAW_ROOT='+str(raw),'-DGARNET_METAL_TEXT_WORKER_EXECUTABLE='+str(n/'xlang3'),'-DGARNET_METAL_PRIMARY_TEST_FAULTS='+('ON'if mode=='fault'else'OFF')]+['-D'+k+'='+('ON'if v else'OFF')for k,v in values.items()]
 with(w/('gateway-build-'+mode+'.log')).open('w')as log:
  subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True);subprocess.run(['/opt/homebrew/bin/cmake','--build',str(b),'-j','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('GATEWAY BUILD PASS '+mode,flush=True)
assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in m.items());print('Gateway source unchanged across requested builds',flush=True)
