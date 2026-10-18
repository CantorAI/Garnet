"""All original native suites, full streams, and bounded observer transactions."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=load(w/'protocol.json');roles=protocol['roles']
assert load(w/'native-preparation.json')['original_legacy_reference_source_exact']
assert load(w/'build-terminal.json')['all_six_builds_pass']
assert not (w/'native-results.json').exists(),'Retain each attempt; never overwrite native results'
assert sha(n/'libgarnet.dylib')==sha(w/'before-libgarnet.dylib')=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
assert not os.environ.get('GARNET_METAL_COUNTER_TRACE')
sources={}
for family in ['legacy','current']:
    s=r/('out/research/worldsense-paired-gpu-profile-'+family)/'source';m=load(w/(family+'-source-manifest.json'))
    assert all(sha(s/k)==v for k,v in m.items());sources[family]=dict(root=str(s),files=m)
paths=[p for p in w.iterdir() if p.is_file() and (p.suffix in ['.py','.cpp','.cmake'] or p.name.endswith('-probe'))]
paths += [w/name for name in ['before-libgarnet.dylib','cases.json','protocol.json','build-plan.json','build-terminal.json','native-preparation.json','native-design-before-implementation.txt','legacy-source-manifest.json','current-source-manifest.json','legacy-source-transformations.json','current-source-transformations.json']]
paths += [n/'xlang3',n/'libxlang3_runtime.dylib',n/'libgarnet.dylib']
paths += [w.parent/'worldsense-opt'/name for name in ['design_paired_counter_ws76.py','setup_paired_counter_ws76.py','prepare_native_ws76.py','verify_native_ws76.py']]
paths += [w.parent/'worldsense-minimal-selective-defaults'/name for name in ['actual-on-main-program.metal','actual-on-prefill-program.metal']]
paths += [r/'out/build/garnet-metal-worldsense-minimal-selective-defaults-on'/name for name in ['primary_projection_words.bin','primary_core_projection_words.bin']]
paths += [Path(k) for k in protocol['evidence']]
inventories={}
for role in roles:
    family,mode=role.split('-');build=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role)
    paths += [p for p in build.iterdir() if p.is_file() and (p.name.startswith('garnet_') or p.name in ['libgarnet.dylib','libgarnet_metal.a','CMakeCache.txt'])]
    tests=json.loads(subprocess.check_output(['/opt/homebrew/bin/ctest','--test-dir',str(build),'--show-only=json-v1']))['tests']
    names=[t['name'] for t in tests]
    baseline=r/('out/build/garnet-metal-worldsense-prefix-cache' if family=='legacy' else 'out/build/garnet-metal-worldsense-minimal-selective-defaults-on')
    original=json.loads(subprocess.check_output(['/opt/homebrew/bin/ctest','--test-dir',str(baseline),'--show-only=json-v1']))['tests']
    assert names==[t['name'] for t in original]+['metal_gpu_pass_counter_reference']
    assert len(tests)==(3 if family=='legacy' else 10)
    inventories[role]=tests
frozen={str(p):sha(p) for p in paths}
assert all(Path(k).name!='worldsense-http.py' for k in frozen)
(w/'native-identities-before-dispatch.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=frozen,sources=sources,test_inventories=inventories,scope='Exact native inputs and binaries, source closures, all controllers/probes. No learned assets consumed. Private authentication helper excluded.'),indent=2)+'\n')
results={};words={};programs={};counter_records={};job='gui/501/com.cantorai.cantor'
try:
    subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False)
    subprocess.run(['launchctl','bootout',job],check=False)
    for _ in range(200):
        if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
        time.sleep(.1)
    else:raise RuntimeError('Normal service did not unload')
    for role in roles:
        family,mode=role.split('-');build=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role)
        with (w/('native-'+role+'.log')).open('w') as log:
            completed=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'],stdout=log,stderr=subprocess.STDOUT,timeout=600)
        results[role]=dict(exit_code=completed.returncode,tests=len(inventories[role]));(w/'native-results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(role,'CTest',len(inventories[role]),'exit',completed.returncode,flush=True)
        assert completed.returncode==0
        if family=='legacy':
            output=w/(role+'-projection-words.bin')
            with (w/('projection-'+role+'.log')).open('w') as log:
                subprocess.run([str(w/(role+'-projection-probe')),str(output)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        else:output=build/'primary_projection_words.bin'
        words[role]=output.read_bytes()
        with (w/('program-'+role+'.log')).open('w') as log:
            subprocess.run([str(w/(role+'-program-probe')),str(w/('actual-'+role))],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        programs[role]={kind:(w/('actual-'+role+'.'+kind+'.metal')).read_bytes() for kind in ['main','prefill']}
        prior=w.parent/'worldsense-minimal-selective-defaults'
        assert programs[role]['main']==(prior/'actual-on-main-program.metal').read_bytes()
        assert programs[role]['prefill']==((prior/'actual-on-prefill-program.metal').read_bytes() if family=='current' else b'')
        trace=build/'gpu-pass-counter-test.jsonl'
        if mode=='quiet':assert not trace.exists();counter_records[role]=dict(commands=0,intervals=0)
        else:
            assert trace.exists();data=trace.read_bytes();(w/('native-counter-'+role+'.jsonl')).write_bytes(data)
            commands=[];pending=[];records=0
            for line in data.decode().splitlines():
                prefix,value=line.split(' ',1);v=json.loads(value)
                assert v['diagnostic'] is True and v['serialized'] is False
                if prefix=='metal_counter_profile':
                    assert v['index']==len(pending) and 0<v['start_raw']<=v['end_raw']<2**64-1
                    pending.append(v);records+=1
                else:
                    assert prefix=='metal_counter_command' and len(pending)==v['samples']==2 and v['partial_submissions']==0 and v['gpu_span_ms']>=0
                    assert v['decode_step'] in [4,16,28]
                    for sample in pending:
                        assert all(sample[k]==v[k] for k in ['pid','context','command','request','model','decode_step'])
                    commands.append(v);pending=[]
            assert not pending and len(commands)==(7 if mode=='fault' else 4) and records==2*len(commands)
            counter_records[role]=dict(commands=len(commands),intervals=records)
        print(role,'original186records/programs/counter transactions pass',flush=True)
    reference=(r/'out/build/garnet-metal-worldsense-minimal-selective-defaults-on/primary_projection_words.bin').read_bytes()
    assert all(data==reference for data in words.values())
    for role in ['current-quiet','current-trace','current-fault']:
        build=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role)
        assert (build/'primary_core_projection_words.bin').read_bytes()==(r/'out/build/garnet-metal-worldsense-minimal-selective-defaults-on/primary_core_projection_words.bin').read_bytes()
        summary=load(build/'primary_core_projection_words.bin.summary.json')
        assert summary['complete_records']==20 and summary['complete_FP32_words']==432804 and summary['core_enabled'] is False
        assert all(summary[k]==0 for k in ['core_decode_delta','core_prefill_delta','core_head_delta'])
    assert all(sha(Path(k))==v for k,v in frozen.items())
    for group in sources.values():assert all(sha(Path(group['root'])/k)==v for k,v in group['files'].items())
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_original_native_suites_plus_observer_pass=True,native_CTest_count=sum(v['tests'] for v in results.values()),roles=results,full_original_projection_records=186,all_six_complete_projection_streams_equal_seed74=True,projection_bytes=len(reference),projection_sha256=hashlib.sha256(reference).hexdigest(),current_original20records432804words_exact=True,programs={role:{kind:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for kind,data in group.items()} for role,group in programs.items()},legacy_actual_MAIN_exact_current_original_MAIN=True,legacy_prefill_empty_current_prefill_exact_seed74=True,counter_transactions=counter_records,input_identities_unchanged=True,no_learned_or_timing_execution=True)
    (w/'native-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    print('NATIVE39_CTEST_ORIGINAL186_COMPLETE_STREAMS_AND_CURRENT20_COMPLETE_STREAMS_ACTUAL_PROGRAMS_COUNTER_TRANSACTIONS_PASS',flush=True)
finally:
    subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
