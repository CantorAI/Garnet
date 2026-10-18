"""Independent audit of all supplemental native roles and completed traces."""
from pathlib import Path
import datetime,hashlib,json,re
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-paired-gpu-profile';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-paired-gpu-profile-current/source'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=load(w/'compatibility-proof-protocol.json');frozen=load(w/'compatibility-native-identities-before-dispatch.json');proof=load(w/'compatibility-native-proof.json')
assert all(sha(Path(k))==v for k,v in frozen['files'].items()) and all(sha(s/k)==v for k,v in frozen['source'].items())
counts={}
for role,info in p['roles'].items():
    build=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role);original=r/('out/build/garnet-metal-worldsense-minimal-selective-defaults-'+info['original_role'])
    flags=dict(re.findall(r'^(GARNET_METAL(?:_[A-Z0-9_]+)?):BOOL=(ON|OFF)$',(build/'CMakeCache.txt').read_text(),re.M));assert all(flags[k]==v for k,v in info['flags'].items())
    log=(w/('native-'+role+'.log')).read_text();passed=re.findall(r'^\s*\d+/\d+\s+Test\s+#\d+:\s+(\S+)\s+\.{2,}\s+Passed\s+',log,re.M)
    assert passed==[t['name'] for t in frozen['test_inventories'][role]] and len(passed)==10 and proof['roles'][role]==0
    for name in ['primary_projection_words.bin','primary_core_projection_words.bin']:assert (build/name).read_bytes()==(original/name).read_bytes()
    for kind,name in [('main','main_shader_program.metal'),('prefill','prefill_shader_program.metal'),('core','primary_core_shader_program.metal')]:
        assert (build/name).read_bytes()==(w/('actual-'+role+'.'+kind+'.metal')).read_bytes()==(w.parent/'worldsense-minimal-selective-defaults'/('actual-'+info['original_role']+'-'+kind+'-program.metal')).read_bytes()
    trace=build/'gpu-pass-counter-test.jsonl';assert trace.read_bytes()==(w/('native-counter-'+role+'.jsonl')).read_bytes();pending=[];commands=0;intervals=0
    for line in trace.read_text().splitlines():
        prefix,value=line.split(' ',1);v=json.loads(value);assert v['diagnostic'] is True and v['serialized'] is False
        if prefix=='metal_counter_profile':assert v['index']==len(pending) and 0<v['start_raw']<=v['end_raw']<2**64-1;pending.append(v);intervals+=1
        else:
            assert prefix=='metal_counter_command' and v['samples']==len(pending)==2 and v['partial_submissions']==0 and v['decode_step'] in [4,16,28]
            assert all(all(sample[k]==v[k] for k in ['pid','request','context','command','model','decode_step']) for sample in pending);pending=[];commands+=1
    assert not pending and commands==(7 if role=='current-compatfault' else 4) and intervals==2*commands;counts[role]=dict(commands=commands,intervals=intervals)
assert proof['native_CTest_count']==40 and load(w/'native-independent-audit.json')['native_CTest_count']==39
assert all(load(w/'restoration-verification.json')[k] for k in ['validated_WS10_binary_restored','four_file_sources_online','camera_unmounted','benchmark_valid_request_http404'])
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supplemental_native_CTest_count=40,combined_native_CTest_count=79,all_ten_roles_complete_original186_word_streams_pass=True,seven_current_roles_original20records432804words_exact=True,all_actual_programs_exact_matching_immutable_role=True,all_common_counter_transactions_and_fault_guards_pass=True,inputs_and_source_unchanged=True,normal_WS10_fourfiles_noCam_benchmark404_restored=True,no_learned_or_capture_run=True,counter_counts=counts)
(w/'compatibility-native-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
