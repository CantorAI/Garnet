"""Original seven configuration rejection cases plus all new packed dependencies."""
from pathlib import Path
import ast,json,subprocess,datetime,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-packed-request-residency';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-packed-request-residency/source';load=lambda p:json.loads(p.read_text())
assert load(w/'learned-independent-audit.json')['old_raw_policy_and_new_packed_lifecycle_pass']
old=w.parent/'worldsense-request-weight-residency/prepare_admission.py';tree=ast.parse(old.read_text());loop=next(v for v in tree.body if isinstance(v,ast.For)and isinstance(v.target,ast.Tuple)and [n.id for n in v.target.elts]==['name','flags','expected']);cases=ast.literal_eval(loop.iter);assert len(cases)==7
base=['/opt/homebrew/bin/cmake','-S',str(s),'-C',str(w/'build-profile/MetalWorldSense4BPrefix.cmake'),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DXLANG3_SDK_ROOT='+str(r/'xlang3/sdk'),'-DGARNET_XLANG3_RUNTIME_LIBRARY='+str(r/'out/build/macos-arm64-release/bin/libxlang3_runtime.dylib'),'-DGARNET_METAL_PRIMARY_BF16_BLOCKS=ON','-DGARNET_METAL_WIDE_PRIMARY_STORAGE=ON','-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=ON','-DGARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY=OFF','-DGARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS=OFF','-DGARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY=OFF','-DGARNET_METAL_PACKED_REQUEST_RESIDENCY_TEST_FAULTS=OFF']
message='Packed request residency requires primary/wide/lazy and excludes raw request residency';wide='Wide primary storage profile requires primary weights and global lazy defaults'
cases += [('packed-primary',['-DGARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY=ON','-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF'],wide),('packed-wide',['-DGARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY=ON','-DGARNET_METAL_WIDE_PRIMARY_STORAGE=OFF'],message),('packed-lazy',['-DGARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY=ON','-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=OFF'],wide),('packed-and-raw',['-DGARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY=ON','-DGARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY=ON'],message),('packed-fault-without-policy',['-DGARNET_METAL_PACKED_REQUEST_RESIDENCY_TEST_FAULTS=ON'],'Packed request test hooks require packed request residency')]
results={}
for name,flags,expected in cases:
 # Original seven cases use their original default wide=OFF; otherwise a newly enabled
 # wide precheck could obscure an unrelated original dependency diagnostic.
 neutral=['-DGARNET_METAL_WIDE_PRIMARY_STORAGE=OFF']if name in {v[0]for v in ast.literal_eval(loop.iter)}else []
 command=base+['-B',str(w/('invalid-'+name+'-config'))]+neutral+flags
 p=subprocess.run(command,capture_output=True,text=True);text=p.stdout+p.stderr;(w/('invalid-'+name+'-config.log')).write_text(text)
 assert p.returncode!=0 and expected in ' '.join(text.split()),(name,p.returncode,text[-1500:]);results[name]=dict(command=command,returncode=p.returncode,expected_diagnostic=expected)
(w/'flag-dependency-proof.json').write_text(json.dumps(results,indent=2)+'\n')
(w/'admission-complete.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_expected_config_rejections=len(results)==12,original7_configurations_retained=True,five_new_invalid_dependency_combinations_rejected=True,original_driver_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),no_performance_result=True),indent=2)+'\n')
print('WS69 original7 plus5 new configuration dependency combinations rejected as required')
