"""Five sequential isolated builds; no GPU/model execution."""
from pathlib import Path
import subprocess,json,hashlib,sys,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-gpu-counter-profile/source'
m=json.loads((w/'correctness-source-manifest.json').read_text());assert len(m)==1720
assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in m.items())
flags=['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']
(w/'build-driver-identity.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(m),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
for mode in sys.argv[1:]or['off','primary','on','quiet','fault']:
 assert mode in ['off','primary','on','quiet','fault'];b=r/('out/build/garnet-metal-worldsense-gpu-counter-profile-'+mode);primary=mode!='off';vector=mode in ['on','quiet','fault'];values={key:(primary if i<3 else vector)for i,key in enumerate(flags)}
 command=['/opt/homebrew/bin/cmake','-S',str(s),'-B',str(b),'-C',str(w/'build-profile/MetalWorldSense4BPrefix.cmake'),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DXLANG3_SDK_ROOT='+str(r/'xlang3/sdk'),'-DGARNET_XLANG3_RUNTIME_LIBRARY='+str(r/'out/build/macos-arm64-release/bin/libxlang3_runtime.dylib'),'-DGARNET_METAL_GPU_PASS_COUNTER_PROFILE='+('OFF'if mode=='quiet'else'ON'),'-DGARNET_METAL_PRIMARY_TEST_FAULTS='+('ON'if mode=='fault'else'OFF')]+['-D'+key+'='+('ON'if value else'OFF')for key,value in values.items()]
 with(w/('build-'+mode+'.log')).open('w')as log:
  subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['/opt/homebrew/bin/cmake','--build',str(b),'-j','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('BUILD PASS '+mode,flush=True)
assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in m.items())
print('Five build source identities unchanged',flush=True)
