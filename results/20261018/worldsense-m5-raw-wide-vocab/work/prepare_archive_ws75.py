"""Prepare archive-only reconstruction of all WS75 source, proofs and rejection."""
from pathlib import Path
import ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');f=b/'work/worldsense-opt'
def adapt(t):return t.replace('worldsense-minimal-selective-defaults','worldsense-raw-wide-vocab').replace('worldsense-m5-minimal-selective-defaults','worldsense-m5-raw-wide-vocab').replace('WS-M5-74','WS-M5-75').replace('WS74','WS75').replace('ws74','ws75')
t=adapt((f/'archive_ws74.py').read_text()).replace("w.parent/'worldsense-minimal-full-prefill'","w.parent/'worldsense-minimal-selective-defaults'")
t=t.replace("['primary_projection_words.bin','primary_core_projection_words.bin','primary_core_projection_words.bin.summary.json']","['primary_projection_words.bin','primary_core_projection_words.bin','primary_core_projection_words.bin.summary.json','raw_wide_vocab_words.bin','raw_wide_vocab_words.bin.summary.json']")
t=t.replace("'primary_corpus_test','direct_float_corpus_test','core_library_corpus_test'","'primary_corpus_test','raw_wide_corpus_test','direct_float_corpus_test','core_library_corpus_test'")
t=t.replace('source_files=1721','source_files=1723').replace('Valid rejection: BOTHmodels derived generation rates fail unchanged2%gates despite favorable observedlatencies/wholeSkill. No later stages/promotion.','Valid rejection:4Btotal/4Bdecode/textTTFT/texttotal/wholeSkill fail unchanged2%gates. Textdecode preserved; no later stages/promotion.')
t=t.replace('core_rows=[];attempts=[]','core_rows=[];head_rows=[];attempts=[]')
needle="    core_rows.append(dict(identity,counter_telemetry_available=available,**derived))";assert t.count(needle)==1;t=t.replace(needle,needle+"\n    head_rows.append(dict(identity,counter_telemetry_available=available,raw_wide_vocab_method='primary_raw_wide_vocab'in c['backend'],active_packed_owners=t['primary_storage_after']['active_weights']if available else None,body_encoded_delta=t['primary_encoded_delta']['decode_encoded']if available else None,packed_head_encoded_delta=t['primary_encoded_delta']['head_encoded']if available else None))")
t=t.replace("('core-dispatch-deltas.csv',core_rows),","('core-dispatch-deltas.csv',core_rows),('vocabulary-storage-policy.csv',head_rows),")
t=t.replace('core_delta_rows=len(core_rows),counter_available_rows=','core_delta_rows=len(core_rows),vocabulary_policy_rows=len(head_rows),counter_available_rows=')
ast.parse(t);p=f/'archive_ws75.py';assert not p.exists();p.write_text(t)
v=adapt((f/'verify_archive_ws74.py').read_text());v=v.replace('len(current)==1721','len(current)==1723').replace("seed_ref='4c811ec4e305317efac622177d4623b1cf8c83a4';seed_prefix='results/20261018/worldsense-m5-minimal-full-prefill/'","seed_ref='e15c3f4325682098bd4c7b4a17da62254e1e36c5';seed_prefix='results/20261018/worldsense-m5-minimal-selective-defaults/'")
a=v.index("changed=['cmake/");z=v.index("assert len(recon['overlay_files'])",a)
sourceproof='''changed=['cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/primary_wide_vocab_policy.h','src/entry/metal_main.cpp','test2026/metal/raw_wide_vocab_test.cpp']
assert sorted(k for k in current if current[k]!=seed_source.get(k))==changed
assert set(current)-set(seed_source)=={'src/backends/metal/primary_wide_vocab_policy.h','test2026/metal/raw_wide_vocab_test.cpp'}
operations=load(d/'source-transformations.json')
for rel in set(c['path']for c in operations):
 text=current[rel].decode()
 for change in reversed([c for c in operations if c['path']==rel]):
  if change.get('append'):assert text.endswith(change['after']);text=text[:-len(change['after'])]
  else:assert text.count(change['after'])==1;text=text.replace(change['after'],change['before'])
 assert text.encode()==seed_source[rel]
assert load(d/'source-inverse-proof.json')['changed_or_added']==changed
repair=load(d/'pre-native-fixture-repair.json');text=current[repair['path']].decode();assert text.count(repair['after'])==1
assert hashlib.sha256(text.encode()).hexdigest()==repair['after_sha256']and hashlib.sha256(text.replace(repair['after'],repair['before']).encode()).hexdigest()==repair['before_sha256']
initial=load(d/'pre-native-fixture-review/correctness-source-manifest.json');final=load(d/'correctness-source-manifest.json');assert initial.keys()==final.keys()and[k for k in final if final[k]!=initial[k]]==[repair['path']]
assert sha(d/'pre-native-fixture-review/raw_wide_vocab_test.cpp')==initial[repair['path']]==repair['before_sha256']
assert repair['observed_initial_build_exit']==0 and repair['all_six_initial_builds_pass']and repair['no_native_model_or_timing_attempt_yet']and repair['only_new_fixture_coverage_changed']
for name in ['pre-native-fixture-review/build-controller.log','rebuild-controller.log']:assert(d/name).read_text().splitlines()==['BUILD PASS '+role for role in ['off','primary','on','fault','compat56','compat56-fault']]
''';v=v[:a]+sourceproof+v[z:];v=v.replace("len(recon['overlay_files'])==16","len(recon['overlay_files'])==18")
v=v.replace(r'\d+/9\s+Test',r'\d+/10\s+Test').replace('list(range(1,10))','list(range(1,11))').replace('out of 9','out of 10')
x=" assert flags['PRIMARY_ONLY_LAZY_DEFAULTS']==('ON'if selective else'OFF')";assert v.count(x)==1;v=v.replace(x,x+"\n assert flags['PRIMARY_RAW_WIDE_VOCAB']==('ON'if role in ['on','fault']else'OFF')")
native='''reference=None;raw_summaries={};raw_cases=[(0,65536),(1,17),(127,129),(65536-33,33),(64,128),(0,4)]
diagnostic=current['src/backends/metal/metal_context.mm'].decode().split('R"RAWGPU(',1)[1].split(')RAWGPU"',1)[0].encode()
for role in roles:
 data=(d/('native-'+role+'-raw_wide_vocab_words.bin')).read_bytes()
 if reference is None:reference=data
 assert data==reference;position=0;total=0
 for start,count in raw_cases:
  assert struct.unpack('<2Q',data[position:position+16])==(start,count);position+=16
  expected=struct.pack('<'+'I'*count,*[(i%65280 if i%65280<32640 else i%65280+128)<<16 for i in range(start,start+count)])
  assert data[position:position+count*4]==expected;position+=count*4;total+=count
 assert position==len(data)and total==65847
 summary=load(d/('native-'+role+'-raw_wide_vocab_words.bin.summary.json'));raw_summaries[role]=summary
 assert summary['pure_policy_boundary_cases']==192 and summary['complete_records']==6 and summary['complete_FP32_words']==65847 and summary['original_HOST_words']==65536 and summary['all_finite_BF16_patterns']==65280
 assert summary['new_policy_enabled']==(role in ['on','fault'])and summary['invalid_cases_rejected']==(6 if role=='off'else 7)and summary['same_device_other_context_pass']and summary['owner_accounting_restored']
 assert(d/('actual-'+role+'-rawGPU-program.metal')).read_bytes()==diagnostic
rawproof=load(d/'rawGPU-native-word-comparison.json');assert rawproof['records']==6 and rawproof['words_per_build']==65847 and rawproof['bytes']==len(reference)and rawproof['sha256']==hashlib.sha256(reference).hexdigest()and rawproof['summaries']==raw_summaries
assert rawproof['diagnostic_source_bytes']==len(diagnostic)and rawproof['diagnostic_source_sha256']==hashlib.sha256(diagnostic).hexdigest()
'''
x="for name,info in load(d/'learned-guard-adaptation-proof.json').items():";assert v.count(x)==1;v=v.replace(x,native+x)
v=v.replace("value.replace('worldsense-raw-wide-vocab','worldsense-minimal-full-prefill')==original","value.replace('worldsense-raw-wide-vocab','worldsense-minimal-selective-defaults')==original")
a=v.index('# Recursively prove');z=v.index('def canonical_learned_checks():',a)
chain='''# Recover every unchanged numerical guard through75->74->71->70->56.
for child_ref,child_prefix,child_slug,parent_ref,parent_prefix,parent_slug in [
 (seed_ref,seed_prefix,'worldsense-minimal-selective-defaults','4c811ec4e305317efac622177d4623b1cf8c83a4','results/20261018/worldsense-m5-minimal-full-prefill/','worldsense-minimal-full-prefill'),
 ('4c811ec4e305317efac622177d4623b1cf8c83a4','results/20261018/worldsense-m5-minimal-full-prefill/','worldsense-minimal-full-prefill','80787c104fdafe7eb4671cc6c7b390bfcce2fb1f','results/20261018/worldsense-m5-primary-core-isolation/','worldsense-primary-core-isolation'),
 ('80787c104fdafe7eb4671cc6c7b390bfcce2fb1f','results/20261018/worldsense-m5-primary-core-isolation/','worldsense-primary-core-isolation','290f5742ba51c41197f24179cfee4876e96e789a','results/20261017/worldsense-m5-request-weight-residency/','worldsense-request-weight-residency')]:
 parent_proof=json.loads(batch([child_ref+':'+child_prefix+'learned-guard-adaptation-proof.json'])[0])
 for name,info in parent_proof.items():
  value=batch([child_ref+':'+child_prefix+name])[0].decode();original=batch([parent_ref+':'+parent_prefix+name])[0].decode()
  assert hashlib.sha256(value.encode()).hexdigest()==info['current_sha256']and hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']
  for op in reversed(info['operations']):assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
  assert value.replace(child_slug,parent_slug)==original
''';v=v[:a]+chain+v[z:]
# Materialize the complete already-frozen original+supplemental learned checks
# against archive paths, not current local binaries or mutable runtime results.
outer=(f/'verify_learned_ws75.py').read_text();a=outer.index("terminal=load(w/'learned-controller-terminal.json')");z=outer.index('result=dict(',a);body=outer[a:z]
for before,after in [("w.parent/'worldsense-primary-blocks/matrix-selection.json'","d/'frozen-native-guards/matrix-selection.json'"),("w.parent/'worldsense-prefix-cache/state-guard.json'","d/'frozen-native-guards/state-guard.json'"),("w.parent/'worldsense-prefix-cache/prefix-model-guard.json'","d/'frozen-native-guards/prefix-model-guard.json'"),("w.parent/'worldsense-request-residency/vision-grid-off.json'","d/'frozen-native-guards/vision-grid-original.json'")]:body=body.replace(before,after)
body=body.replace('w/','d/')
builder='''resolution_for_builders=load(d/'external-input-resolution.json');root=repo.parent
for filename,role,cpp,exe in [('corpus-build-identity.json','compat56','primary_corpus_test.cpp','primary_corpus_test'),('raw-wide-corpus-build-identity.json','on','raw_wide_corpus_test.cpp','raw_wide_corpus_test'),('direct-corpus-build-identity.json','compat56','direct_float_corpus_test.cpp','direct_float_corpus_test')]:
 record=load(d/filename);build=root/('out/build/garnet-metal-worldsense-raw-wide-vocab-'+role);command=record['command'];original_cpp=b/'work/worldsense-raw-wide-vocab'/cpp
 assert record['source_sha256']==sha(d/cpp)and record['library_sha256']==resolution_for_builders[str(build/'libgarnet_metal.a')]['sha256']and record['executable_sha256']==resolution_for_builders[str(build/exe)]['sha256']
 assert str(original_cpp)in command and str(build/'libgarnet_metal.a')in command and command[-2:]==['-o',str(build/exe)]and ('-DGARNET_METAL_PRIMARY_RAW_WIDE_VOCAB=1'in command)==(role=='on')
'''
function='def canonical_learned_checks():\n'+''.join(' '+line+'\n'for line in(builder+body).splitlines())
a=v.index('def canonical_learned_checks():');z=v.index('canonical_learned_checks()\n',a+len('def canonical_learned_checks():'));v=v[:a]+function+'\n'+v[z:]
v=v.replace("len(load(d/'flag-dependency-proof.json'))==18","len(load(d/'flag-dependency-proof.json'))==20").replace('==22792','==22942').replace("['files'])==1875","['files'])==1880").replace('native_CTest_pass=54','native_CTest_pass=60').replace('source_files=1721','source_files=1723').replace('offline_malformed_copies_rejected=22792','offline_malformed_copies_rejected=22942')
x="  assert all(t['primary_encoded_delta'][key]>0 for key in ['decode_encoded','prefill_encoded','head_encoded'])";y="  assert t['primary_encoded_delta']['decode_encoded']>0 and t['primary_encoded_delta']['prefill_encoded']>0 and t['primary_encoded_delta']['head_encoded']==t['primary_encoded_delta']['vector_head_encoded']==0 and t['primary_storage_after']['active_weights']==200";assert v.count(x)==1;v=v.replace(x,y)
newcsv='''head_csv=list(csv.DictReader((d/'vocabulary-storage-policy.csv').open()));assert len(head_csv)==32
lookup_head={(v['stage'],v['worker'],int(v['repetition']),v['phase']):v for v in head_csv}
for stage,label,index,phase,item,call in rawcalls:
 row=lookup_head[(stage,label,index,phase)];t=call['timings'];present='primary_storage_after'in t
 assert row['counter_telemetry_available']==str(present)and row['raw_wide_vocab_method']==str('primary_raw_wide_vocab'in call['backend'])
 for field,value in [('active_packed_owners',t['primary_storage_after']['active_weights']if present else None),('body_encoded_delta',t['primary_encoded_delta']['decode_encoded']if present else None),('packed_head_encoded_delta',t['primary_encoded_delta']['head_encoded']if present else None)]:
  assert row[field]==(''if value is None else str(value))
assert scope['vocabulary_policy_rows']==32
''';x="assert load(d/'restoration-verification.json')['benchmark_valid_request_http404']";assert v.count(x)==1;v=v.replace(x,newcsv+x)
v=v.replace('exact57_three_policy_imports_inverse_to71=True','sevenpath_inverse_to74_and_new_semantic_policy_rawGPU_diagnostic_proven=True,new6complete_rawGPUrecords65847words_all65280finite_patterns_exact=True').replace('original_guard74_to71_to70_to56_and_all_five_metric_gate_ASTs_inverse_exact=True','original_guard75_to74_to71_to70_to56_plus_supplemental_corpus_and_five_metric_ASTs_inverse_exact=True').replace('original_five_full_learned_readers_in_actual_candidate_states_tokens_vision_lifecycle_exact=True','originalCOMPAT201_five_GPU_readers_each4200857600words=True,actualcandidate200_five_GPU_readers_each3565158400words_plus635699200rawheadGPUwords_and_all_original_states_tokens_vision_lifecycle_exact=True')
ast.parse(v);p=f/'verify_archive_ws75.py';assert not p.exists();p.write_text(v)
print('WS75 archive/canonical verifier ready: sevenpath74inverse/fixturehistory/native60/rawGPUwords/201COMPAT+200candidate/fullrawheadGPU/guardchain/20configs/22942negativecopies/32pipelinecalls andpolicyCSV. No copies/inference yet.')
