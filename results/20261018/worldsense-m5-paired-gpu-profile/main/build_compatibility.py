"""Sequential four additional original compatibility builds, without GPU work."""
from pathlib import Path
import datetime,hashlib,json,subprocess
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-paired-gpu-profile-current/source'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=load(w/'compatibility-proof-protocol.json');m=load(w/'current-source-manifest.json')
assert sha(w/'full-proof-addendum-before-supplemental-build.txt')==p['design_sha256'] and all(sha(s/k)==v for k,v in m.items())
plan={};inputs={str(Path(__file__)):sha(Path(__file__)),str(w/'compatibility-proof-protocol.json'):sha(w/'compatibility-proof-protocol.json')}
for role,info in p['roles'].items():
    original=Path(info['original_cache']);assert sha(original)==info['original_cache_sha256'];inputs[str(original)]=sha(original)
    build=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role);assert not build.exists()
    cmd=['/opt/homebrew/bin/cmake','-S',str(s),'-B',str(build),'-C',str(w/'build-profile/MetalWorldSense4BPrefix.cmake'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DXLANG3_SDK_ROOT='+str(r/'xlang3/sdk'),'-DGARNET_XLANG3_RUNTIME_LIBRARY='+str(r/'out/build/macos-arm64-release/bin/libxlang3_runtime.dylib')]+['-D'+k+'='+v for k,v in sorted(info['flags'].items())]
    plan[role]=[cmd,['/opt/homebrew/bin/cmake','--build',str(build),'-j','2']]
(w/'compatibility-build-plan.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=inputs,commands=plan),indent=2)+'\n')
for role,commands in plan.items():
    with (w/('build-'+role+'.log')).open('w') as log:
        for command in commands:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('BUILD PASS',role,flush=True)
assert all(sha(s/k)==v for k,v in m.items()) and all(sha(Path(k))==v for k,v in inputs.items())
(w/'compatibility-build-terminal.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_four_compatibility_builds_pass=True,inputs_unchanged=True,no_native_model_or_capture_execution=True),indent=2)+'\n')
print('Supplemental four original-proof builds complete, identities unchanged; no native/model/capture execution',flush=True)
