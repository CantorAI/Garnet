from pathlib import Path
import subprocess,sys,hashlib,json,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-simd-metadata/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(s/k)==v for k,v in load(w/'correctness-source-manifest.json').items())
commands={}
for role in ['off','primary','on','fault','compat67']:
 primary='OFF'if role=='off'else'ON';full='ON'if role=='compat67'else'OFF';meta='ON'if role in ['on','fault']else'OFF';build=r/('out/build/garnet-metal-worldsense-primary-simd-metadata-'+role)
 flags={'GARNET_METAL_PRIMARY_BF16_BLOCKS':primary,'GARNET_METAL_WIDE_PRIMARY_STORAGE':primary,'GARNET_METAL_LAZY_RESIDENT_DEFAULTS':primary,'GARNET_METAL_PRIMARY_SIMD_METADATA':meta,'GARNET_METAL_PRIMARY_TEST_FAULTS':'ON'if role=='fault'else'OFF'}
 flags.update({k:full for k in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_SOLE_INTERLEAVED_MLP','GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL','GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE','GARNET_METAL_SOLE_COOPERATIVE_PREFILL','GARNET_METAL_CACHED_MATRIX_METADATA']})
 command=['/opt/homebrew/bin/cmake','-S',str(s),'-B',str(build),'-C',str(w/'build-profile/MetalWorldSense4BPrefix.cmake'),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DXLANG3_SDK_ROOT='+str(r/'xlang3/sdk'),'-DGARNET_XLANG3_RUNTIME_LIBRARY='+str(r/'out/build/macos-arm64-release/bin/libxlang3_runtime.dylib')]+['-D'+k+'='+v for k,v in flags.items()]
 commands[role]=[command,['/opt/homebrew/bin/cmake','--build',str(build),'-j','2']]
(w/'build-plan.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands=commands,driver_sha256=sha(Path(__file__))),indent=2)+'\n')
for role in sys.argv[1:]or commands:
 assert role in commands
 with(w/('build-'+role+'.log')).open('w')as log:
  for command in commands[role]:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('BUILD PASS',role,flush=True)
