from pathlib import Path
import subprocess,sys,json,hashlib,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-minimal-full-prefill/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=load(w/'correctness-source-manifest.json');assert len(manifest)==1721 and all(sha(s/k)==v for k,v in manifest.items())
roles=['off','primary','on','fault','compat56','compat56-fault'];plan={}
for role in roles:
 primary='OFF'if role=='off'else'ON';compat=role.startswith('compat56');full='ON'if compat else'OFF';packed='ON'if role in ['on','fault']else'OFF';build=r/('out/build/garnet-metal-worldsense-minimal-full-prefill-'+role)
 flags={'GARNET_METAL_PRIMARY_BF16_BLOCKS':primary,'GARNET_METAL_WIDE_PRIMARY_STORAGE':primary,'GARNET_METAL_LAZY_RESIDENT_DEFAULTS':primary,'GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY':packed,'GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY':'ON'if compat else'OFF','GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS':'ON'if role=='compat56-fault'else'OFF','GARNET_METAL_PRIMARY_TEST_FAULTS':'ON'if role in ['fault','compat56-fault']else'OFF'}
 flags['GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY']='OFF'
 flags.update({key:'ON'if compat or role in ['on','fault']else'OFF'for key in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']})
 flags.update({key:full for key in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_CACHED_MATRIX_METADATA']})
 configure=['/opt/homebrew/bin/cmake','-S',str(s),'-B',str(build),'-C',str(w/'build-profile/MetalWorldSense4BPrefix.cmake'),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DXLANG3_SDK_ROOT='+str(r/'xlang3/sdk'),'-DGARNET_XLANG3_RUNTIME_LIBRARY='+str(r/'out/build/macos-arm64-release/bin/libxlang3_runtime.dylib')]+['-D'+key+'='+value for key,value in flags.items()]
 plan[role]=[configure,['/opt/homebrew/bin/cmake','--build',str(build),'-j','2']]
(w/'build-plan.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands=plan,driver_sha256=sha(Path(__file__))),indent=2)+'\n')
for role in sys.argv[1:]or roles:
 assert role in plan
 with(w/('build-'+role+'.log')).open('w')as log:
  for command in plan[role]:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('BUILD PASS',role,flush=True)
