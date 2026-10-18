"""Canonical CORE verifier: original native/learned/metric proofs, no inference."""
from pathlib import Path
import ast,textwrap
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');f=b/'work/worldsense-opt';prior=(f/'verify_archive_ws69.py').read_text();text=prior[:prior.index('delta=sorted(')].replace('worldsense-m5-packed-request-residency','worldsense-m5-primary-core-isolation')
text+='''delta=sorted(k for k,v in current.items()if seed_source.get(k)!=v)
assert delta==sorted(['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/primary_core_isolation_test.cpp'])
source=load(d/'correctness-source-manifest.json');transforms=load(d/'source-transformations.json')
for rel,operations in transforms.items():
 value=current[rel].decode()
 for op in reversed(operations):
  if op.get('append'):assert value.endswith(op['after']);value=value[:-len(op['after'])]
  else:assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
 assert value.encode()==seed_source[rel],rel
assert all(current[k]==seed_source[k]for k in seed if k not in transforms)
roles=['off','primary','on','fault','compat56','compat56-fault'];assert load(d/'native-results.json')=={role:0 for role in roles}
import re,struct,copy
for role in roles:
 log=(d/('native-'+role+'.log')).read_text();assert sorted(int(v)for v in re.findall(r'\\d+/9\\s+Test\\s+#\\s*(\\d+):[^\\n]*Passed',log))==list(range(1,10))and re.search(r'100% tests passed(?:, 0 tests failed)? out of 9',log)
 cache=(d/'build-identities'/role/'CMakeCache.txt').read_text();assert ('GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY:BOOL=ON'in cache)==(role in ['on','fault'])
 assert ('GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY:BOOL=ON'in cache)==role.startswith('compat56')
 assert ('GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS:BOOL=ON'in cache)==(role=='compat56-fault')
raw=(d/'native-off-primary_projection_words.bin').read_bytes();assert raw==batch([seed_ref+':'+seed_prefix+'native-off-projection-words.bin'])[0]
assert all((d/('native-'+role+'-primary_projection_words.bin')).read_bytes()==raw for role in roles)
proof=load(d/'native-word-comparison.json');assert proof['records']==186 and proof['bytes']==len(raw)and proof['sha256']==hashlib.sha256(raw).hexdigest()
reference=None;cases=[(rows,inner,columns,0)for inner in [128,256]for columns in [260,1024]for rows in [1,2,16,32]]+[(rows,128,100352,0)for rows in [1,2]]+[(rows,128,260,1)for rows in [1,2]]
for role in roles:
 data=(d/('native-'+role+'-primary_core_projection_words.bin')).read_bytes()
 if reference is None:reference=data
 assert data==reference;position=0;words=0
 for rows,inner,columns,raw_mode in cases:
  count=rows*columns;assert struct.unpack('<6Q',data[position:position+48])==(rows,inner,columns,raw_mode,int(rows<=2 and columns<100000),count);position+=48
  expected=b''.join(struct.pack('<f',inner*(row+1)/16)*columns for row in range(rows));assert data[position:position+count*4]==expected;position+=count*4;words+=count
 assert position==len(data)and words==432804
 summary=load(d/('native-'+role+'-primary_core_projection_words.bin.summary.json'));assert summary['complete_records']==20 and summary['complete_FP32_words']==432804 and summary['specialized_body_cases']==10 and summary['core_enabled']==(role in ['on','fault'])
 assert [summary[k]for k in ['core_decode_delta','core_prefill_delta','core_head_delta']]==([8,8,2]if role in ['on','fault']else[0,0,0])
proof=load(d/'core-native-word-comparison.json');assert proof['records']==20 and proof['words_per_build']==432804 and proof['bytes']==len(reference)and proof['sha256']==hashlib.sha256(reference).hexdigest()
programs={role:{kind:(d/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()for kind in ['main','prefill','core']}for role in roles}
for role,previous in [('off','off'),('primary','primary'),('compat56','on'),('compat56-fault','fault')]:
 for kind in ['main','prefill']:assert programs[role][kind]==batch([seed_ref+':'+seed_prefix+'actual-'+previous+'-'+kind+'-program.metal'])[0]
core_program=current['src/backends/metal/primary_isolation_shader.h'].decode().split('R"IPRIMARY(',1)[1].split(')IPRIMARY"',1)[0]+current['src/backends/metal/primary_block_shader.h'].decode().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]
for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and not programs[role]['prefill']and programs[role]['core']==core_program.encode()
assert all(not programs[role]['core']for role in ['off','primary','compat56','compat56-fault'])
for role,groups in load(d/'actual-program-proof.json')['programs'].items():
 for kind,info in groups.items():assert len(programs[role][kind])==info['bytes']and hashlib.sha256(programs[role][kind]).hexdigest()==info['sha256']
for name,info in load(d/'learned-guard-adaptation-proof.json').items():
 original=batch([seed_ref+':'+seed_prefix+name])[0].decode();value=(d/name).read_text();assert hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']and hashlib.sha256(value.encode()).hexdigest()==info['current_sha256']
 for op in reversed(info['operations']):assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
 assert value.replace('worldsense-primary-core-isolation','worldsense-request-weight-residency')==original
'''
learned=(f/'verify_learned_ws70.py').read_text();core=learned[learned.index("terminal=load(w/'learned-controller-terminal.json')"):learned.index('result=dict(')]
core=core.replace("w.parent/'worldsense-primary-blocks/matrix-selection.json'","d/'frozen-native-guards/matrix-selection.json'").replace("w.parent/'worldsense-prefix-cache/state-guard.json'","d/'frozen-native-guards/state-guard.json'").replace("w.parent/'worldsense-prefix-cache/prefix-model-guard.json'","d/'frozen-native-guards/prefix-model-guard.json'").replace("w.parent/'worldsense-request-residency/vision-grid-off.json'","d/'frozen-native-guards/vision-grid-original.json'").replace('load(w/','load(d/').replace('(w/','(d/')
text+='def canonical_learned_checks():\n'+textwrap.indent(core,' ')+'\ncanonical_learned_checks()\n'
section=prior[prior.index("assert load(d/'native-independent-audit.json')['observed_exit_code']"):prior.index("gold=load(d/'oracles/worldsense-golden.json')")]
section=section.replace("len(load(d/'flag-dependency-proof.json'))==12","len(load(d/'flag-dependency-proof.json'))==15").replace('worldsense-packed-request-residency','worldsense-primary-core-isolation').replace('audit_routes_ws69.py','audit_routes_ws70.py').replace('==8149','==21122')
text+=section
tail=prior[prior.index("gold=load(d/'oracles/worldsense-golden.json')"):prior.index('cleanup=list(csv.DictReader')]
text+=tail
text+='''core_rows=list(csv.DictReader((d/'core-dispatch-deltas.csv').open()));assert len(core_rows)==32
lookup={(v['stage'],v['worker'],int(v['repetition']),v['phase']):v for v in core_rows};available=0;positive_vlm=0
for stage,label,index,phase,item,call in rawcalls:
 t=call['timings'];folder=d/'production-speed'if stage=='production-speed'else d/'functional-smoke'if stage=='functional-correctness-only'else d
 assert load(folder/(label+'.json'))['request']==load(folder/'cases.json')
 cohort=next(v for v in load(folder/'cohort-manifest.json')if v['label']==label);environment=load((d/'production-speed/environment.json')if stage=='production-speed'else d/'environment.json');enabled=label=='candidate-1'
 assert cohort['client_exit']==0 and cohort['library_sha256']==environment['libraries']['on'if enabled else'off']
 assert cohort['serving_sha256']==(load(d/'correctness-source-manifest.json')['runtime/metal_serving.py']if enabled or stage!='production-speed'else environment['control_serving_sha256'])
 row=lookup[(stage,label,index,phase)];present='primary_storage_before'in t;assert row['counter_telemetry_available']==str(present)
 if present:available+=1
 for key,oldkey in [('core_library_decode_encoded','decode_encoded'),('core_library_prefill_encoded','prefill_encoded'),('core_library_head_encoded','head_encoded')]:
  if not present:assert row[key]=='';continue
  actual=t['primary_storage_after'][key]-t['primary_storage_before'][key];assert int(row[key])==actual==(t['primary_encoded_delta'][oldkey]if enabled else 0)
 if present and enabled and call['model_id']=='Qwen3.5-4B':positive_vlm+=1
assert available==22 and positive_vlm==9
scope=load(d/'export-scope.json');assert scope['core_delta_rows']==32 and scope['counter_available_rows']==22 and scope['memory_rows']==22 and scope['calls']==32
assert load(d/'restoration-verification.json')['benchmark_valid_request_http404']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_files=len(manifest),source_files=1721,source_overlays=len(recon['overlay_files']),legacy_files=1706,legacy_overlays=len(legacy_recon['overlay_files']),native_CTest_pass=54,original186_complete_projection_streams_and_new20records432804words_exact=True,original_programs_and_runtime_shader_math_CUDA_engine_test_bytes_preserved=True,original_two_program_contract_and_separate_core_getter_partition_exact=True,original_guard_and_all_five_metric_gate_ASTs_inverse_exact=True,original_five_full_learned_readers_and_two_CORE_readers_states_tokens_vision_lifecycle_exact=True,offline_malformed_copies_rejected=21122,all32_pipeline_goldens_native_routes_CSV_resource_speed_and_rejection_recomputed=True,calls=32,requests=16,memory_rows=22,core_delta_rows=32,core_counter_available_rows=22,positive_CORE_VLM_calls=9,external_input_resolutions=len(resolution),decision='valid-reject',promotion=False)
(b/'work/worldsense-opt/ws70-archive-independent-audit.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result,indent=2))
'''
ast.parse(text);p=f/'verify_archive_ws70.py';assert not p.exists();p.write_text(text);print('WS70 canonical source/native/CORE/learned/32call metric verifier prepared, no inference')
