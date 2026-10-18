"""Six sequential isolated diagnostic builds. No GPU/model tests here."""
from pathlib import Path
import datetime, hashlib, json, re, subprocess, sys
w=Path(__file__).resolve().parent
r=Path('/Users/shawn/CantorAI')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((w/'protocol.json').read_text())
assert sha(w/'native-design-before-implementation.txt')==protocol['design_sha256']
roles=protocol['roles']
cache_paths={
 'legacy':r/'out/build/garnet-metal-worldsense-prefix-cache/CMakeCache.txt',
 'current':r/'out/build/garnet-metal-worldsense-minimal-selective-defaults-on/CMakeCache.txt',
}
flags={}
for family,path in cache_paths.items():
    flags[family]=dict(re.findall(r'^(GARNET_METAL(?:_[A-Z0-9_]+)?):BOOL=(ON|OFF)$',path.read_text(),re.M))
    assert flags[family]['GARNET_METAL']=='ON' and flags[family]['GARNET_METAL_XMODEL']=='ON'
    assert flags[family]['GARNET_METAL_QWEN35_4B_WIDE_PREFILL']=='ON'
    assert flags[family]['GARNET_METAL_DENSE_PREFIX_CACHE']=='ON'
current_only=[k for k,v in flags['current'].items() if v=='ON' and flags['legacy'].get(k,'OFF')!='ON']
assert set(current_only)=={'GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS'},current_only
assert all(flags['current'].get(k)==v for k,v in flags['legacy'].items())
plan={};frozen={str(Path(__file__)):sha(Path(__file__)),str(w/'protocol.json'):sha(w/'protocol.json')}
for path in list(cache_paths.values())+list((w/'build-profile').glob('*.cmake'))+[w/'native-design-before-implementation.txt']:
    frozen[str(path)]=sha(path)
for role in roles:
    family,mode=role.split('-');s=r/('out/research/worldsense-paired-gpu-profile-'+family)/'source'
    manifest=json.loads((w/(family+'-source-manifest.json')).read_text())
    assert len(manifest)==(1708 if family=='legacy' else 1723) and all(sha(s/k)==v for k,v in manifest.items())
    values=dict(flags[family]);values['GARNET_METAL_GPU_PASS_COUNTER_PROFILE']='OFF' if mode=='quiet' else 'ON'
    values['GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS']='ON' if mode=='fault' else 'OFF'
    if family=='current': values['GARNET_METAL_PRIMARY_TEST_FAULTS']='ON' if mode=='fault' else 'OFF'
    b=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role)
    assert not b.exists()
    configure=['/opt/homebrew/bin/cmake','-S',str(s),'-B',str(b),'-C',str(w/'build-profile/MetalWorldSense4BPrefix.cmake'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DXLANG3_SDK_ROOT='+str(r/'xlang3/sdk'),'-DGARNET_XLANG3_RUNTIME_LIBRARY='+str(r/'out/build/macos-arm64-release/bin/libxlang3_runtime.dylib')]+['-D'+key+'='+value for key,value in sorted(values.items())]
    plan[role]=dict(flags=values,commands=[configure,['/opt/homebrew/bin/cmake','--build',str(b),'-j','2']])
record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),driver_sha256=sha(Path(__file__)),frozen_inputs=frozen,commands=plan,no_native_or_model_tests=True)
(w/'build-plan.json').write_text(json.dumps(record,indent=2)+'\n')
for role in roles:
    with (w/('build-'+role+'.log')).open('w') as log:
        for command in plan[role]['commands']:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('BUILD PASS',role,flush=True)
for family in ['legacy','current']:
    s=r/('out/research/worldsense-paired-gpu-profile-'+family)/'source'
    manifest=json.loads((w/(family+'-source-manifest.json')).read_text())
    assert all(sha(s/k)==v for k,v in manifest.items())
assert all(sha(Path(k))==v for k,v in frozen.items())
(w/'build-terminal.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),roles=roles,all_six_builds_pass=True,source_and_driver_identities_unchanged=True,no_native_or_model_tests=True),indent=2)+'\n')
print('All six builds complete; source/driver identities unchanged; no native/model run',flush=True)
