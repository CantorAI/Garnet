"""Keep original complete native proofs and add the bounded raw GPU word proof."""
from pathlib import Path
import ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-raw-wide-vocab';old=w.parent/'worldsense-minimal-selective-defaults';f=b/'work/worldsense-opt'
t=(old/'native_controller.py').read_text().replace('worldsense-minimal-selective-defaults','worldsense-raw-wide-vocab').replace('len(manifest)==1721','len(manifest)==1723').replace("w/'build-controller.log'","w/'rebuild-controller.log'").replace('Native9tests','Native10tests').replace('ALL6ROLES9TESTS','ALL6ROLES10TESTS')
t=t.replace("'immutable57-policy-guard.txt',","'prospective-byte-budget.json','pre-native-fixture-repair.json','raw_wide_vocab_test.cpp',")
x="['design_selective_ws74.py','setup_selective_ws74.py','prepare_native_ws74.py','verify_native_ws74.py']";assert t.count(x)==1;t=t.replace(x,"['design_raw_wide_vocab_ws75.py','setup_raw_wide_vocab_ws75.py','repair_native_fixture_ws75.py','prepare_native_ws75.py','verify_native_ws75.py']")
newproof='''
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
  program=(build/'raw_bf16_gpu_program.metal').read_bytes();assert program==diagnostic;(w/('actual-'+role+'-rawGPU-program.metal')).write_bytes(program)
 (w/'rawGPU-native-word-comparison.json').write_text(json.dumps(dict(records=6,words_per_build=65847,all_six_complete_streams_equal=True,all_original_BF16_bit_references_exact=True,all65280finite_patterns_covered=True,bytes=len(reference),sha256=hashlib.sha256(reference).hexdigest(),summaries=raw_summaries,diagnostic_source_bytes=len(diagnostic),diagnostic_source_sha256=hashlib.sha256(diagnostic).hexdigest(),scope='Pure192semantic cases plus bounded original raw GPU word proof; not an actual learned vocabulary/serving/speed result.'),indent=2)+'\\n')
'''
anchor=" assert all(sha(s/k)==v for k,v in manifest.items())and all(sha(Path(k))==v for k,v in frozen.items())";assert t.count(anchor)==1;t=t.replace(anchor,newproof+anchor);ast.parse(t);(w/'native_controller.py').write_text(t)
v=(f/'verify_native_ws74.py').read_text().replace('worldsense-minimal-selective-defaults','worldsense-raw-wide-vocab');start=v.index('assert len(source)==');end=v.index('assert all(sha(Path(k))',start)
block='''assert len(source)==1723 and len(seed)==1721 and all(sha(s/k)==value for k,value in source.items())
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
''';v=v[:start]+block+v[end:]
v=v.replace(r'\d+/9\s+Test',r'\d+/10\s+Test').replace('list(range(1,10))','list(range(1,11))').replace('out of 9','out of 10')
proof='\n'.join(line[1:]if line.startswith(' ')else line for line in newproof.splitlines())
proof=proof.replace(";(w/('actual-'+role+'-rawGPU-program.metal')).write_bytes(program)",";assert(w/('actual-'+role+'-rawGPU-program.metal')).read_bytes()==program")
start=proof.index("(w/'rawGPU-native-word-comparison.json')");proof=proof[:start]+'''rawproof=load(w/'rawGPU-native-word-comparison.json');assert rawproof['records']==6 and rawproof['words_per_build']==65847 and rawproof['bytes']==len(reference)and rawproof['sha256']==hashlib.sha256(reference).hexdigest()and rawproof['summaries']==raw_summaries
assert rawproof['diagnostic_source_bytes']==len(diagnostic)and rawproof['diagnostic_source_sha256']==hashlib.sha256(diagnostic).hexdigest()
'''
anchor="assert load(w/'restoration-verification.json')['benchmark_valid_request_http404']";assert v.count(anchor)==1;v=v.replace(anchor,proof+'\n'+anchor)
v=v.replace('source_files=1721,exact57_three_file_policy_import_and_inverse71=True,six_role_selective_and_compatibility_flags_exact=True,native_tests_pass=54','source_files=1723,sevenpath_inverse_to74_and_original_production_programs_exact=True,six_role_new_rawwide_and_selective_compatibility_flags_exact=True,native_tests_pass=60,new_rawGPU6records65847words_and_all65280finite_BF16_patterns_exact=True,pure_semantic_boundary_cases=192')
ast.parse(v);(f/'verify_native_ws75.py').write_text(v)
print('WS75 native60: original186+20complete streams/programs, plus192purepolicy/6rawGPUrecords65847exactwords. No native/model dispatched; newfixture repair preserved.')
