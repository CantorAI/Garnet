"""Independently audit native outputs against frozen inputs and both seeds."""
from pathlib import Path
import datetime, hashlib, json, re
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-paired-gpu-profile';r=Path('/Users/shawn/CantorAI')
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frozen=load(w/'native-identities-before-dispatch.json');protocol=load(w/'protocol.json');proof=load(w/'native-proof.json')
assert all(sha(Path(k))==v for k,v in frozen['files'].items())
assert sha(w/'native-design-before-implementation.txt')==protocol['design_sha256']
expected=['cmake/GarnetMetal.cmake','runtime/metal_serving.py','src/backends/metal/gpu_pass_counter_trace.inc','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/gpu_pass_counter_test.cpp']
helpers=[];tests=[]
for family in ['legacy','current']:
    group=frozen['sources'][family];s=Path(group['root']);seed=load(w/(family+'-seed-source-manifest.json'));manifest=group['files']
    assert len(manifest)==(1708 if family=='legacy' else 1723) and all(sha(s/k)==v for k,v in manifest.items())
    assert sorted(k for k,v in manifest.items() if seed.get(k)!=v)==expected
    for rel,ops in load(w/(family+'-source-transformations.json')).items():
        text=(s/rel).read_text()
        for op in reversed(ops):
            if op.get('append'):assert text.endswith(op['after']);text=text[:-len(op['after'])]
            else:assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
        assert hashlib.sha256(text.encode()).hexdigest()==seed[rel]
    helpers.append((s/'src/backends/metal/gpu_pass_counter_trace.inc').read_text());tests.append((s/'test2026/metal/gpu_pass_counter_test.cpp').read_text())
assert helpers[0]==helpers[1]==(b/'work/worldsense-gpu-counter-profile/gpu_pass_counter_trace.inc').read_text().replace('GARNET_METAL_PRIMARY_TEST_FAULTS','GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS')
assert tests[0]==tests[1]==(b/'work/worldsense-gpu-counter-profile/gpu_pass_counter_test.cpp').read_text().replace('GARNET_METAL_PRIMARY_TEST_FAULTS','GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS')
preparation=load(w/'native-preparation.json')
assert all(v['diagnostic_label_arguments_and_wrapper_absent'] and v['original_encoder_call_present'] for v in preparation['preprocessing'].values())
assert sha(w/'legacy_projection_test.cpp')==sha(b/'work/worldsense-validated-text-worker/legacy_projection_test.cpp')
reference=(r/'out/build/garnet-metal-worldsense-minimal-selective-defaults-on/primary_projection_words.bin').read_bytes()
roles=protocol['roles'];count=0;counter_counts={}
for role in roles:
    family,mode=role.split('-');build=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role)
    cache=dict(re.findall(r'^(GARNET_METAL(?:_[A-Z0-9_]+)?):BOOL=(ON|OFF)$',(build/'CMakeCache.txt').read_text(),re.M))
    wanted=load(w/'build-plan.json')['commands'][role]['flags'];assert all(cache[k]==v for k,v in wanted.items())
    assert cache['GARNET_METAL_GPU_PASS_COUNTER_PROFILE']==('OFF' if mode=='quiet' else 'ON')
    assert cache['GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS']==('ON' if mode=='fault' else 'OFF')
    if family=='current':assert cache['GARNET_METAL_PRIMARY_TEST_FAULTS']==('ON' if mode=='fault' else 'OFF')
    log=(w/('native-'+role+'.log')).read_text();number=3 if family=='legacy' else 10
    assert '100% tests passed, 0 tests failed out of '+str(number) in log
    assert len(frozen['test_inventories'][role])==number and proof['roles'][role]==dict(exit_code=0,tests=number)
    count+=number
    actual=(w/(role+'-projection-words.bin') if family=='legacy' else build/'primary_projection_words.bin').read_bytes();assert actual==reference
    assert (w/('actual-'+role+'.main.metal')).read_bytes()==(b/'work/worldsense-minimal-selective-defaults/actual-on-main-program.metal').read_bytes()
    assert (w/('actual-'+role+'.prefill.metal')).read_bytes()==((b/'work/worldsense-minimal-selective-defaults/actual-on-prefill-program.metal').read_bytes() if family=='current' else b'')
    if family=='current':
        assert (build/'primary_core_projection_words.bin').read_bytes()==(r/'out/build/garnet-metal-worldsense-minimal-selective-defaults-on/primary_core_projection_words.bin').read_bytes()
    trace=build/'gpu-pass-counter-test.jsonl'
    if mode=='quiet':assert not trace.exists();counter_counts[role]=dict(commands=0,intervals=0);continue
    assert trace.read_bytes()==(w/('native-counter-'+role+'.jsonl')).read_bytes()
    pending=[];commands=0;intervals=0
    for line in trace.read_text().splitlines():
        prefix,value=line.split(' ',1);v=json.loads(value);assert v['diagnostic'] is True and v['serialized'] is False
        if prefix=='metal_counter_profile':
            assert 0<v['start_raw']<=v['end_raw']<2**64-1 and v['index']==len(pending);pending.append(v);intervals+=1
        else:
            assert prefix=='metal_counter_command' and v['samples']==len(pending)==2 and v['partial_submissions']==0
            assert all(all(p[k]==v[k] for k in ['pid','context','command','request','model','decode_step']) for p in pending)
            assert v['decode_step'] in [4,16,28];commands+=1;pending=[]
    assert not pending and commands==(7 if mode=='fault' else 4) and intervals==commands*2
    counter_counts[role]=dict(commands=commands,intervals=intervals)
assert count==39 and proof['native_CTest_count']==39 and proof['projection_sha256']==hashlib.sha256(reference).hexdigest() and proof['projection_bytes']==len(reference)
assert proof['counter_transactions']==counter_counts
normal=load(w/'restoration-verification.json')
assert all(normal[k] for k in ['validated_WS10_binary_restored','four_file_sources_online','camera_unmounted','benchmark_valid_request_http404'])
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_CTest_count=count,all_six_roles_complete_original186_projection_streams_exact=True,current_original20records432804words_exact=True,all_actual_programs_exact_seed74_with_legacy_prefill_empty=True,two_source_sevenpath_inverse_and_common_counter_helper_test_exact=True,both_quiet_label_preprocessor_proofs_pass=True,all_counter_transactions_and_roles_audited=True,inputs_unchanged=True,normal_WS10_fourfiles_noCam_benchmark404_restored=True,no_learned_or_timing_run=True)
(w/'native-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
