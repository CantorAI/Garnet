"""Independent post-terminal source/word/program/counter audit, no model/timing."""
from pathlib import Path
import json,hashlib,re,struct,sys,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-raw-wide-vocab';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-raw-wide-vocab/source';old=r/'out/research/worldsense-request-weight-residency/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();roles=['off','primary','on','fault','compat56','compat56-fault'];source=load(w/'correctness-source-manifest.json');seed=load(w/'seed-source-manifest.json')
assert len(source)==1723 and len(seed)==1721 and all(sha(s/k)==value for k,value in source.items())
published=json.loads(__import__('subprocess').check_output(['git','-C',str(r/'Garnet'),'show','e15c3f4325682098bd4c7b4a17da62254e1e36c5:results/20261018/worldsense-m5-minimal-selective-defaults/correctness-source-manifest.json']))
assert seed==published
expected=['cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/primary_wide_vocab_policy.h','src/entry/metal_main.cpp','test2026/metal/raw_wide_vocab_test.cpp']
assert sorted(k for k in source if source[k]!=seed.get(k))==expected
ops=load(w/'source-transformations.json')
for rel in set(x['path']for x in ops):
 text=(s/rel).read_text()
 for op in reversed([x for x in ops if x['path']==rel]):
  if op['append']:assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==1;text=text.replace(op['after'],op['before'])
 assert hashlib.sha256(text.encode()).hexdigest()==published[rel]
assert set(source)-set(seed)=={'src/backends/metal/primary_wide_vocab_policy.h','test2026/metal/raw_wide_vocab_test.cpp'}
assert load(w/'source-inverse-proof.json')['changed_or_added']==expected
repair=load(w/'pre-native-fixture-repair.json');text=(s/repair['path']).read_text();assert text.count(repair['after'])==1 and hashlib.sha256(text.replace(repair['after'],repair['before']).encode()).hexdigest()==repair['before_sha256']
assert hashlib.sha256(text.encode()).hexdigest()==repair['after_sha256']and repair['observed_initial_build_exit']==0 and repair['no_native_model_or_timing_attempt_yet']
for role in roles:
 cache=(r/('out/build/garnet-metal-worldsense-raw-wide-vocab-'+role)/'CMakeCache.txt').read_text();flags=dict(re.findall(r'^GARNET_METAL_([A-Z0-9_]+):BOOL=(ON|OFF)$',cache,re.M));selective=role in ['primary','on','fault'];compat=role.startswith('compat56');vector=role in ['on','fault']or compat
 assert flags['PRIMARY_RAW_WIDE_VOCAB']==('ON'if role in ['on','fault']else'OFF')and flags['PRIMARY_ONLY_LAZY_DEFAULTS']==('ON'if selective else'OFF')
 for key in ['PRIMARY_BF16_BLOCKS','WIDE_PRIMARY_STORAGE','LAZY_RESIDENT_DEFAULTS']:assert flags[key]==('OFF'if role=='off'else'ON')
 for key in ['PRIMARY_VECTOR_PREFILL','PRIMARY_PREFILL_ISOLATED_LIBRARY','PRIMARY_FULL_ISOLATED_LIBRARY']:assert flags[key]==('ON'if vector else'OFF')
 for key in ['PRIMARY_VECTOR_PREFILL_K32_ONLY','DIRECT_FLOAT_WEIGHT_UPLOAD','CACHED_MATRIX_METADATA','REQUEST_SCOPE_WEIGHT_RESIDENCY']:assert flags[key]==('ON'if compat else'OFF')
 assert flags['PRIMARY_CORE_ISOLATED_LIBRARY']=='OFF'and flags['PRIMARY_TEST_FAULTS']==('ON'if role in ['fault','compat56-fault']else'OFF')and flags['REQUEST_RESIDENCY_TEST_FAULTS']==('ON'if role=='compat56-fault'else'OFF')
assert all(sha(Path(k))==v for k,v in load(w/'native-input-identities-before-dispatch.json')['files'].items())
assert load(w/'native-results.json')=={role:0 for role in roles}
for role in roles:
 text=(w/('native-'+role+'.log')).read_text();assert sorted(int(v)for v in re.findall(r'\d+/10\s+Test\s+#\s*(\d+):[^\n]*Passed',text))==list(range(1,11))and re.search(r'100% tests passed(?:, 0 tests failed)? out of 10',text)
raw=(r/'out/build/garnet-metal-worldsense-raw-wide-vocab-off/primary_projection_words.bin').read_bytes();assert raw==(r/'out/build/garnet-metal-worldsense-request-weight-residency-off/primary_projection_words.bin').read_bytes()
assert all((r/('out/build/garnet-metal-worldsense-raw-wide-vocab-'+role)/'primary_projection_words.bin').read_bytes()==raw for role in roles)
proof=load(w/'native-word-comparison.json');assert proof['records']==186 and proof['bytes']==len(raw)and proof['sha256']==hashlib.sha256(raw).hexdigest()
reference=None;cases=[(rows,inner,columns,0)for inner in [128,256]for columns in [260,1024]for rows in [1,2,16,32]]+[(rows,128,100352,0)for rows in [1,2]]+[(rows,128,260,1)for rows in [1,2]]
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-raw-wide-vocab-'+role);data=(build/'primary_core_projection_words.bin').read_bytes()
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

reference=None;raw_summaries={};raw_cases=[(0,65536),(1,17),(127,129),(65536-33,33),(64,128),(0,4)]
diagnostic=(s/'src/backends/metal/metal_context.mm').read_text().split('R"RAWGPU(',1)[1].split(')RAWGPU"',1)[0].encode()
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-raw-wide-vocab-'+role);data=(build/'raw_wide_vocab_words.bin').read_bytes()
 if reference is None:reference=data
 assert data==reference;position=0;total=0
 for start,count in raw_cases:
  assert struct.unpack('<2Q',data[position:position+16])==(start,count);position+=16
  bits=[(i%65280 if i%65280<32640 else i%65280+128)<<16 for i in range(start,start+count)];expected=struct.pack('<'+'I'*count,*bits)
  assert data[position:position+count*4]==expected;position+=count*4;total+=count
 assert position==len(data)and total==65847
 summary=load(build/'raw_wide_vocab_words.bin.summary.json');raw_summaries[role]=summary
 assert summary['pure_policy_boundary_cases']==192 and summary['complete_records']==6 and summary['complete_FP32_words']==65847 and summary['original_HOST_words']==65536 and summary['all_finite_BF16_patterns']==65280
 assert summary['new_policy_enabled']==(role in ['on','fault'])and summary['invalid_cases_rejected']==(6 if role=='off'else 7)and summary['same_device_other_context_pass']and summary['owner_accounting_restored']
 program=(build/'raw_bf16_gpu_program.metal').read_bytes();assert program==diagnostic;assert(w/('actual-'+role+'-rawGPU-program.metal')).read_bytes()==program
rawproof=load(w/'rawGPU-native-word-comparison.json');assert rawproof['records']==6 and rawproof['words_per_build']==65847 and rawproof['bytes']==len(reference)and rawproof['sha256']==hashlib.sha256(reference).hexdigest()and rawproof['summaries']==raw_summaries
assert rawproof['diagnostic_source_bytes']==len(diagnostic)and rawproof['diagnostic_source_sha256']==hashlib.sha256(diagnostic).hexdigest()

assert load(w/'restoration-verification.json')['benchmark_valid_request_http404']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_native_session=int(sys.argv[1]),observed_exit_code=0,source_files=1723,sevenpath_inverse_to74_and_original_production_programs_exact=True,six_role_new_rawwide_and_selective_compatibility_flags_exact=True,native_tests_pass=60,new_rawGPU6records65847words_and_all65280finite_BF16_patterns_exact=True,pure_semantic_boundary_cases=192,original186_complete_projection_records_and_FP64_bounds_preserved=True,new20_complete_records432804words_exact_all_six=True,all_actual_core_route_counters_and_getter_zero=True,actual_FULL_MAIN_exact_RAW_and_secondary_exact_original_fragments=True,original_two_program_vector_prefill_contract_and_all_compat56_programs_preserved=True,source_and_all_explicit_native_inputs_binaries_unchanged=True,no_model_or_speed_result_yet=True,promotion=False)
(w/'native-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
