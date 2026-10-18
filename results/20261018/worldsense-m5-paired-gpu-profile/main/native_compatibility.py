"""Retain all original current native proof routes before learned execution."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time,shutil
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';s=r/'out/research/worldsense-paired-gpu-profile-current/source'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=load(w/'compatibility-proof-protocol.json');m=load(w/'current-source-manifest.json')
assert load(w/'compatibility-build-terminal.json')['all_four_compatibility_builds_pass']
assert not(w/'compatibility-native-results.json').exists()
assert sha(n/'libgarnet.dylib')==sha(w/'before-libgarnet.dylib')=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
assert all(sha(s/k)==v for k,v in m.items())
files=[w/name for name in ['native_compatibility.py','build_compatibility.py','restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib','compatibility-proof-protocol.json','full-proof-addendum-before-supplemental-build.txt','compatibility-build-plan.json','compatibility-build-terminal.json','current-source-manifest.json','current-seed-source-manifest.json','current-source-transformations.json']]
files += [w.parent/'worldsense-opt'/name for name in ['prepare_compatibility_ws76.py','verify_native_compatibility_ws76.py']]+[n/'xlang3',n/'libxlang3_runtime.dylib',n/'libgarnet.dylib']
inventories={}
for role,info in p['roles'].items():
    build=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role);original=r/('out/build/garnet-metal-worldsense-minimal-selective-defaults-'+info['original_role'])
    files += [q for q in build.iterdir() if q.is_file() and (q.name.startswith('garnet_') or q.name in ['libgarnet.dylib','libgarnet_metal.a','CMakeCache.txt'])]
    files += [original/name for name in ['primary_projection_words.bin','primary_core_projection_words.bin']]
    files += [w.parent/'worldsense-minimal-selective-defaults'/('actual-'+info['original_role']+'-'+kind+'-program.metal') for kind in ['main','prefill','core']]
    tests=json.loads(subprocess.check_output(['/opt/homebrew/bin/ctest','--test-dir',str(build),'--show-only=json-v1']))['tests']
    old=json.loads(subprocess.check_output(['/opt/homebrew/bin/ctest','--test-dir',str(original),'--show-only=json-v1']))['tests']
    assert len(tests)==10 and [t['name'] for t in tests]==[t['name'] for t in old]+['metal_gpu_pass_counter_reference'];inventories[role]=tests
frozen={str(q):sha(q) for q in files}
assert all(Path(k).name!='worldsense-http.py' for k in frozen)
(w/'compatibility-native-identities-before-dispatch.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=frozen,source=m,test_inventories=inventories),indent=2)+'\n')
shutil.copy2(w/'restoration-verification.json',w/'initial-six-native-restoration-verification.json')
results={};job='gui/501/com.cantorai.cantor'
try:
    subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
    for _ in range(200):
        if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
        time.sleep(.1)
    else:raise RuntimeError('Service did not unload')
    for role,info in p['roles'].items():
        build=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role);original=r/('out/build/garnet-metal-worldsense-minimal-selective-defaults-'+info['original_role'])
        with (w/('native-'+role+'.log')).open('w') as log:completed=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'],stdout=log,stderr=subprocess.STDOUT,timeout=600)
        results[role]=completed.returncode;(w/'compatibility-native-results.json').write_text(json.dumps(results,indent=2)+'\n');print(role,'native10 exit',completed.returncode,flush=True);assert completed.returncode==0
        for name in ['primary_projection_words.bin','primary_core_projection_words.bin']:assert (build/name).read_bytes()==(original/name).read_bytes()
        for kind,name in [('main','main_shader_program.metal'),('prefill','prefill_shader_program.metal'),('core','primary_core_shader_program.metal')]:
            data=(build/name).read_bytes();assert data==(w.parent/'worldsense-minimal-selective-defaults'/('actual-'+info['original_role']+'-'+kind+'-program.metal')).read_bytes();(w/('actual-'+role+'.'+kind+'.metal')).write_bytes(data)
        assert not (build/'primary_core_shader_program.metal').read_bytes()
        trace=(build/'gpu-pass-counter-test.jsonl').read_bytes();(w/('native-counter-'+role+'.jsonl')).write_bytes(trace)
        summary=load(build/'primary_core_projection_words.bin.summary.json');assert summary['complete_records']==20 and summary['complete_FP32_words']==432804 and summary['core_enabled'] is False
        assert all(summary[k]==0 for k in ['core_decode_delta','core_prefill_delta','core_head_delta'])
    assert all(sha(Path(k))==v for k,v in frozen.items()) and all(sha(s/k)==v for k,v in m.items())
    (w/'compatibility-native-proof.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_CTest_count=40,all_four_additional_original_routes_pass=True,all_original186_and20_complete_word_streams_exact_each_matching_seed74=True,all_actual_MAIN_prefill_CORE_programs_exact_each_matching_seed74=True,all_inputs_unchanged=True,no_learned_or_capture_run=True,roles=results),indent=2)+'\n')
    print('Four supplemental roles native40/original186+20 streams/programs PASS',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
