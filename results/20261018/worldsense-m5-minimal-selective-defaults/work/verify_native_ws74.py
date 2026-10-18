"""Independent post-terminal source/word/program/counter audit, no model/timing."""
from pathlib import Path
import json,hashlib,re,struct,sys,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-minimal-selective-defaults';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-minimal-selective-defaults/source';old=r/'out/research/worldsense-request-weight-residency/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();roles=['off','primary','on','fault','compat56','compat56-fault'];source=load(w/'correctness-source-manifest.json');seed=load(w/'seed-source-manifest.json')
assert len(source)==len(seed)==1721 and source.keys()==seed.keys()and all(sha(s/k)==value for k,value in source.items())
published=json.loads(__import__('subprocess').check_output(['git','-C',str(r/'Garnet'),'show','4c811ec4e305317efac622177d4623b1cf8c83a4:results/20261018/worldsense-m5-minimal-full-prefill/correctness-source-manifest.json']))
assert published==seed
expected=['cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/entry/metal_main.cpp']
assert sorted(k for k in source if source[k]!=seed[k])==expected
changes=load(w/'source-transformations.json');assert sorted(x['path']for x in changes)==expected
for change in changes:
 text=(s/change['path']).read_text()
 if change.get('append'):assert text.endswith(change['after']);inverse=text[:-len(change['after'])]
 else:assert text.count(change['after'])==1;inverse=text.replace(change['after'],change['before'])
 assert hashlib.sha256(inverse.encode()).hexdigest()==published[change['path']]
policy_source=r/'out/research/worldsense-packed-only-defaults/source';policy_manifest=json.loads(__import__('subprocess').check_output(['git','-C',str(r/'Garnet'),'show','91b1eaded0ce26ad4d03210615626f175f095a1c:results/20261017/worldsense-m5-packed-only-defaults/correctness-source-manifest.json']))
assert all(sha(policy_source/k)==policy_manifest[k]for k in expected)
policy=(policy_source/expected[1]).read_text();a=policy.index('#ifdef GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS',policy.index('void MetalCompiledGraph::RetireResidentDefaults'));z=policy.index('#endif',a)+len('#endif');guard=policy[a:z]+'\n';assert guard==(w/'immutable57-policy-guard.txt').read_text()and(s/expected[1]).read_text().count(guard)==1
cmake=(policy_source/expected[0]).read_text();cmake=cmake[cmake.index('option(GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS'):];assert next(c for c in changes if c['path']==expected[0])['after']=='\n'+cmake
entry=(policy_source/expected[2]).read_text();a=entry.index('#ifdef GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS');z=entry.index('#endif',a)+len('#endif');assert next(c for c in changes if c['path']==expected[2])['after']==entry[a:z]+'\n        return info;\n'
assert load(w/'source-inverse-proof.json')['changed_or_added']==expected
for role in roles:
 cache=(r/('out/build/garnet-metal-worldsense-minimal-selective-defaults-'+role)/'CMakeCache.txt').read_text();flags=dict(re.findall(r'^GARNET_METAL_([A-Z0-9_]+):BOOL=(ON|OFF)$',cache,re.M));selective=role in ['primary','on','fault'];compat=role.startswith('compat56');vector=role in ['on','fault']or compat
 assert flags['PRIMARY_ONLY_LAZY_DEFAULTS']==('ON'if selective else'OFF')
 for key in ['PRIMARY_BF16_BLOCKS','WIDE_PRIMARY_STORAGE','LAZY_RESIDENT_DEFAULTS']:assert flags[key]==('OFF'if role=='off'else'ON')
 for key in ['PRIMARY_VECTOR_PREFILL','PRIMARY_PREFILL_ISOLATED_LIBRARY','PRIMARY_FULL_ISOLATED_LIBRARY']:assert flags[key]==('ON'if vector else'OFF')
 for key in ['PRIMARY_VECTOR_PREFILL_K32_ONLY','DIRECT_FLOAT_WEIGHT_UPLOAD','CACHED_MATRIX_METADATA','REQUEST_SCOPE_WEIGHT_RESIDENCY']:assert flags[key]==('ON'if compat else'OFF')
 assert flags['PRIMARY_CORE_ISOLATED_LIBRARY']=='OFF'
 assert flags['PRIMARY_TEST_FAULTS']==('ON'if role in ['fault','compat56-fault']else'OFF')and flags['REQUEST_RESIDENCY_TEST_FAULTS']==('ON'if role=='compat56-fault'else'OFF')
assert all(sha(Path(k))==v for k,v in load(w/'native-input-identities-before-dispatch.json')['files'].items())
assert load(w/'native-results.json')=={role:0 for role in roles}
for role in roles:
 text=(w/('native-'+role+'.log')).read_text();assert sorted(int(v)for v in re.findall(r'\d+/9\s+Test\s+#\s*(\d+):[^\n]*Passed',text))==list(range(1,10))and re.search(r'100% tests passed(?:, 0 tests failed)? out of 9',text)
raw=(r/'out/build/garnet-metal-worldsense-minimal-selective-defaults-off/primary_projection_words.bin').read_bytes();assert raw==(r/'out/build/garnet-metal-worldsense-request-weight-residency-off/primary_projection_words.bin').read_bytes()
assert all((r/('out/build/garnet-metal-worldsense-minimal-selective-defaults-'+role)/'primary_projection_words.bin').read_bytes()==raw for role in roles)
proof=load(w/'native-word-comparison.json');assert proof['records']==186 and proof['bytes']==len(raw)and proof['sha256']==hashlib.sha256(raw).hexdigest()
reference=None;cases=[(rows,inner,columns,0)for inner in [128,256]for columns in [260,1024]for rows in [1,2,16,32]]+[(rows,128,100352,0)for rows in [1,2]]+[(rows,128,260,1)for rows in [1,2]]
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-minimal-selective-defaults-'+role);data=(build/'primary_core_projection_words.bin').read_bytes()
 if reference is None:reference=data
 assert data==reference;position=0;words=0
 for rows,inner,columns,raw_mode in cases:
  count=rows*columns;assert struct.unpack('<6Q',data[position:position+48])==(rows,inner,columns,raw_mode,int(rows<=2 and columns<100000),count);position+=48
  expected=b''.join(struct.pack('<f',inner*(row+1)/16)*columns for row in range(rows));assert data[position:position+count*4]==expected;position+=count*4;words+=count
 assert position==len(data)and words==432804
 summary=load(build/'primary_core_projection_words.bin.summary.json');assert summary['complete_records']==20 and summary['complete_FP32_words']==432804 and summary['specialized_body_cases']==10 and summary['core_enabled'] is False
 assert [summary[k]for k in ['core_decode_delta','core_prefill_delta','core_head_delta']]==[0,0,0]
new=load(w/'core-native-word-comparison.json');assert new['records']==20 and new['words_per_build']==432804 and new['bytes']==len(reference)and new['sha256']==hashlib.sha256(reference).hexdigest()
programs={role:{kind:(w/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()for kind in ['main','prefill','core']}for role in roles}
for role,previous in [('off','off'),('primary','primary'),('compat56','on'),('compat56-fault','fault')]:
 for kind in ['main','prefill']:assert programs[role][kind]==(w.parent/'worldsense-request-weight-residency'/('actual-'+previous+'-'+kind+'-program.metal')).read_bytes()
full=(s/'src/backends/metal/primary_isolation_shader.h').read_text().split('R"IPRIMARY(',1)[1].split(')IPRIMARY"',1)[0]+(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]+(s/'src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R"PVEC(',1)[1].split(')PVEC"',1)[0]
for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==full.encode()and programs[role]['prefill']==programs['compat56']['prefill']
assert all(not programs[role]['core']for role in roles)
for role,group in load(w/'actual-program-proof.json')['programs'].items():
 for kind,info in group.items():assert len(programs[role][kind])==info['bytes']and hashlib.sha256(programs[role][kind]).hexdigest()==info['sha256']
assert load(w/'restoration-verification.json')['benchmark_valid_request_http404']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_native_session=int(sys.argv[1]),observed_exit_code=0,source_files=1721,exact57_three_file_policy_import_and_inverse71=True,six_role_selective_and_compatibility_flags_exact=True,native_tests_pass=54,original186_complete_projection_records_and_FP64_bounds_preserved=True,new20_complete_records432804words_exact_all_six=True,all_actual_core_route_counters_and_getter_zero=True,actual_FULL_MAIN_exact_RAW_and_secondary_exact_original_fragments=True,original_two_program_vector_prefill_contract_and_all_compat56_programs_preserved=True,source_and_all_explicit_native_inputs_binaries_unchanged=True,no_model_or_speed_result_yet=True,promotion=False)
(w/'native-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
