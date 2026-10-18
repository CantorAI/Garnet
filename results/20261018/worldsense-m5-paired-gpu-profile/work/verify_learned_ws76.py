"""Independent original learned oracles, actual core words/counters and input closure."""
from pathlib import Path
import json,hashlib,datetime,sys
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-paired-gpu-profile-current';old=w.parent/'worldsense-minimal-selective-defaults';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-paired-gpu-profile-current/source';load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
frozen=load(w/'learned-identities-before.json');assert len(frozen['source'])==1723 and all(sha(s/k)==v for k,v in frozen['source'].items())and all(sha(Path(k))==v for k,v in frozen['external'].items())
for name,info in load(w/'learned-guard-adaptation-proof.json').items():
 original=(old/name).read_text();text=(w/name).read_text();assert hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']and hashlib.sha256(text.encode()).hexdigest()==info['current_sha256']
 for op in reversed(info['operations']):assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text.replace('worldsense-paired-gpu-profile-current','worldsense-minimal-selective-defaults')==original
terminal=load(w/'learned-controller-terminal.json');assert terminal['success']and terminal['normal_restored']and not terminal['promotion'];journal=load(w/'learned-controller-journal.json');assert len(journal)==25 and all(v['returncode']==0 and v['ended_utc']>=v['started_utc']for v in journal)
assert all(load(w/'api-correctness-result.json').values());corpus=load(w/'corpus-GPU-bits.json');t=corpus['totals']
assert t['packed_matrices']==201 and t['raw_small_matrices']==245 and t['HOST_raw_FP32_words_checked']==1724383232 and t['retained_primary_and_small_raw_bytes']==9850930176 and t['original_bytes']==11850481664
assert t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600 and t['GPU_prefill_tile_FP32_words_checked']=={k:4200857600 for k in ['32x32','64x32','64x64']}
selection=load(w.parent/'worldsense-primary-blocks/matrix-selection.json')['matrices'];assert len(corpus['matrices'])==446 and {(v['model'],v['tensor'],v['rows'],v['inner'])for v in selection}=={(v['model'],v['tensor'],v['rows'],v['inner'])for v in corpus['matrices']}
for row in corpus['matrices']:
 assert row['owner_accounting_restored'];count=row['rows']*row['inner']
 if row['packed']:assert row['GPU_original_FP32_words_checked']==row['GPU_vector_original_FP32_words_checked']==count and row['GPU_prefill_tile_FP32_words_checked']=={k:count for k in ['32x32','64x32','64x64']}
 else:assert row['HOST_raw_FP32_words_checked']==count
assert load(w/'direct-upload-corpus.json')['totals']==dict(weights=546,FP32_words_checked=334609152)
resident=load(w/'resident-results.json');assert len(resident)==8 and {v['mode']for v in resident}=={'off','primary','on','compat56'}and all(v['returncode']==0 for v in resident);private_steps=0
for row in resident:
 lines=(w/row['log']).read_text().splitlines();steps=[json.loads(v.split(' ',1)[1])for v in lines if v.startswith('private_kv_step ')];assert len(steps)==8 and all(v['exact_output_and_state']for v in steps)and any(v.startswith('private_kv_reference PASS ')for v in lines);private_steps+=len(steps)
for role in ['off','on','fault','compat56']:assert 'Four-input trained linear generic-resident parity/host-private isolation/inactive/malformed/shared-after-source-close/rebuild PASS '+role in(w/('trained-linear-'+role+'.log')).read_text()
vlm=load(w.parent/'worldsense-prefix-cache/state-guard.json')['records'][1];prefix={v['prompt_index']:v for v in load(w.parent/'worldsense-prefix-cache/prefix-model-guard.json')['records']};records={role:load(w/('api-state-progress-'+role+'.json'))['records']for role in ['off','primary','on','quiet','compat56']};core_routes=[]
for role,rows in records.items():
 assert len(rows)==5
 for index,row in enumerate(rows):
  reference=vlm if index<2 else prefix[[0,1,0][index-2]];call=row['call'];timings=call['timings'];assert row['states']==reference['states']and call['generated_token_ids']==reference['generated_token_ids'];baseline=records['off'][index]['call'];assert all(call[k]==baseline[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
  enabled=role!='off';compat=role=='compat56';isolated=False;vector=role in ['on','quiet','compat56']
  for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:assert(cap in call['backend'])==enabled
  for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_full_isolated_library']:assert(cap in call['backend'])==vector
  for cap in ['cached_matrix_metadata','request_scope_weight_residency']:assert(cap in call['backend'])==compat
  assert('primary_core_isolated_library'in call['backend'])==isolated
  selective=role in ['primary','on','quiet'];retirement=enabled and (not selective or index<2)
  assert ('primary_only_lazy_defaults'in call['backend'])==selective
  assert timings['lazy_default_rehydrated_bytes']==0 and(timings['lazy_default_retired_bytes']>0)==retirement and(timings['lazy_default_planned_slots']>0)==retirement
  stats=timings['primary_storage_after'];before=timings['primary_storage_before'];delta=timings['primary_encoded_delta'];derived={}
  for key,oldkey in [('core_library_decode_encoded','decode_encoded'),('core_library_prefill_encoded','prefill_encoded'),('core_library_head_encoded','head_encoded')]:
   assert type(before[key])is int and type(stats[key])is int and 0<=before[key]<=stats[key];derived[oldkey]=stats[key]-before[key];assert derived[oldkey]==(delta[oldkey]if isolated else 0)
   if not isolated:assert before[key]==stats[key]==0
  core_routes.append(dict(role=role,index=index,model=call['model_id'],core_deltas=derived))
  if enabled and index<2:
   assert stats['active_weights']==201 and stats['original_bytes']==8401715200 and stats['payload_bytes']+stats['metadata_bytes']==6402163712 and stats['metadata_bytes']==32819200
   assert all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])and delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']
  else:assert all(v==0 for v in delta.values())and all(v==0 for v in derived.values())
  if compat:
   q=timings['request_weight_residency'];assert q['calls']==q['ends']and q['active_bytes']==q['global_used_bytes']==0 and not q['scope_open']and q['limit_bytes']==4294967296
   if index<2:assert q['reason']==7 and q['admitted']==0 and q['fallback']==q['calls']
   else:assert q['reason']==1 and q['last_count']==197 and q['last_bytes']==3440902144 and q['coverage_fallbacks']==0
for role in ['off','on','compat56']:assert load(w/('vision-grid-'+role+'.json'))==load(w.parent/'worldsense-request-residency/vision-grid-off.json')
assert load(w/'request-session-guard-compat56.json')['all_cases_pass']and load(w/'request-full-session-guard-compat56.json')['real197masters_global4GiB_contention_close_readmission_reset']
assert load(w.parent/'worldsense-paired-gpu-profile/compatibility-native-independent-audit.json')['combined_native_CTest_count']==79

main=w.parent/'worldsense-paired-gpu-profile';legacy=w.parent/'worldsense-paired-gpu-profile-legacy'
assert len(frozen['legacy_source'])==1708 and all(sha(Path(frozen['legacy_root'])/k)==v for k,v in frozen['legacy_source'].items())
assert all(str(Path(k).resolve())==v['resolved']and str(Path(k).readlink())==v['target']and sha(Path(k)/'libgarnet.dylib')==v['library_sha256']for k,v in frozen['build_aliases'].items())
for name,info in load(main/'legacy-and-counter-guard-adaptation-proof.json').items():
 original=Path(info['seed']).read_text();path=Path(info['current_path'])if 'current_path'in info else legacy/name;text=path.read_text()
 assert hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']and hashlib.sha256(text.encode()).hexdigest()==info['current_sha256']
 if info.get('path_replacement_only'):
  family=name.split('-',1)[0];assert text.replace('worldsense-paired-gpu-profile-'+family,'worldsense-gpu-counter-profile')==original
 else:
  for op in reversed(info['operations']):assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
  assert text.replace('worldsense-paired-gpu-profile-legacy','worldsense-minimal-selective-defaults')==original
legacy_records={role:load(legacy/('api-state-progress-'+role+'.json'))['records']for role in ['quiet','trace','fault']}
for role,rows in legacy_records.items():
 assert len(rows)==5 and load(legacy/('api-state-guard-'+role+'.json'))['all32_VLM_and28_text_state_bits_and_tokens_match_previous']
 for index,row in enumerate(rows):
  reference=vlm if index<2 else prefix[[0,1,0][index-2]];call=row['call'];baseline=records['off'][index]['call']
  assert row['states']==reference['states']and call['generated_token_ids']==reference['generated_token_ids']
  assert all(call[k]==baseline[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
  assert call['status']=='ok'and 'primary_bf16_blocks'not in call['backend']and 'lazy_resident_defaults'not in call['backend']
 assert load(legacy/('vision-grid-'+role+'.json'))==load(w.parent/'worldsense-request-residency/vision-grid-off.json')
fault_results={}
for family,folder in [('current',w),('legacy',legacy)]:
 data=load(folder/'counter-fault-model-recovery.json');bad=data['injected_call'];call=data['recovery_call'];baseline=records['off'][2]['call']
 assert bad['status']=='error'and 'Injected GPU trace resolution failure'in bad['error_message']and call['status']=='ok'
 assert data['actual_learned_counter_resolution_error_propagates_and_scope_clears']and data['all28_states_and_tokens_exact_after_reset_recovery']
 assert data['states']==prefix[0]['states']and all(call[k]==baseline[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
 fault_results[family]=data
trace_proofs=[];sequences={}
def audit_trace(path,calls,enabled,label):
 expected={};ids=[]
 for call in calls:
  assert ('gpu_pass_counter_profile'in call['backend'])==enabled
  if not enabled:assert 'counter_trace_request_id'not in call['timings'];continue
  request=call['timings']['counter_trace_request_id'];assert type(request)is int and request>0 and request not in ids;ids.append(request)
  generated=call['generated_tokens'];limit=call['serving_configuration']['requested_output_limit'];steps=max(0,generated-int(generated==limit))
  for step in [4,16,28]:
   if step<=steps:expected[(request,call['model_id'],step)]=call
 if not enabled:
  assert not path.exists();return dict(label=label,commands=0,intervals=0,enabled=False)
 assert path.is_file();pending=[];seen=set();intervals=0;observed=[]
 for line in path.read_text().splitlines():
  tag,payload=line.split(' ',1);v=json.loads(payload);assert v['diagnostic']is True and v['serialized']is False
  if tag=='metal_counter_profile':
   assert v['index']==len(pending)and 0<v['start_raw']<=v['end_raw']<2**64-1;pending.append(v);intervals+=1
  else:
   assert tag=='metal_counter_command'and 0<len(pending)==v['samples']<=2048 and v['partial_submissions']==0 and v['gpu_span_ms']>=0
   key=(v['request'],v['model'],v['decode_step']);assert key in expected and key not in seen;seen.add(key)
   for row in pending:assert all(row[k]==v[k]for k in ['pid','context','command','request','model','decode_step'])
   call=expected[key];seq=[row['op'].removesuffix(':primary')+(':raw'if row['op'].endswith(':primary')else'')for row in pending]
   sequence_key=(call['model_id'],call['input_prompt_sha256'],v['decode_step'])
   if sequence_key in sequences:assert sequences[sequence_key]==seq
   else:sequences[sequence_key]=seq
   observed.append(dict(request=v['request'],model=v['model'],step=v['decode_step'],samples=v['samples'],ordered_normalized_ops_sha256=hashlib.sha256(json.dumps(seq,separators=(',',':')).encode()).hexdigest()));pending=[]
 assert not pending and seen==set(expected)
 return dict(label=label,commands=len(seen),intervals=intervals,enabled=True,file_sha256=sha(path),windows=observed)
for role,rows in records.items():
 trace_proofs.append(audit_trace(w/('api-state-guard-'+role+'.log.counters.jsonl'),[row['call']for row in rows],role!='quiet','current-'+role))
 for row in rows:
  if role!='quiet':assert type(row['call']['timings']['counter_trace_request_id'])is int and row['call']['timings']['counter_trace_request_id']>0
for role,rows in legacy_records.items():
 trace_proofs.append(audit_trace(w/('legacy-api-state-guard-'+role+'.log.counters.jsonl'),[row['call']for row in rows],role!='quiet','legacy-'+role))
for family,data in fault_results.items():trace_proofs.append(audit_trace(w/(family+'-counter-fault-model-recovery.log.counters.jsonl'),[data['recovery_call']],True,family+'-fault-recovery'))
(main/'learned-counter-command-proof.json').write_text(json.dumps(dict(proofs=trace_proofs,all_expected_selected_commands_and_complete_intervals_joined=True,all_aligned_ordered_operations_equal_after_declared_raw_primary_label_normalization=True,no_performance_claim=True),indent=2)+'\n')

result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_terminal_session=int(sys.argv[1]),observed_exit_code=0,source1723_and_all_explicit_inputs_unchanged=True,original_guard_adaptations_inverse_exact=True,original_five_full_GPU_readers_EACH_words=4200857600,all_five_readers_in_actual_FULL_candidate=True,raw_HOST_words=1724383232,direct_words=334609152,resident_private_steps=private_steps,five_current_API_modes25calls_original32VLM28text_states_tokens_vision_exact=True,actual_core_deltas_zero_all_modes=True,selective_raw_text_retirement_zero_and_VLM_positive_all_calls=True,old_raw_compatibility_lifecycles_pass=True,core_route_records=core_routes,no_speed_claim=True,legacy_three_API_modes15calls_and_three_vision_modes_exact=True,both_actual_learned_counter_errors_scope_cleanup_and28state_recovery_exact=True,all_selected_counter_commands_and_aligned_ordered_operation_sequences_audited=True,no_actual_WorldSense_capture=True,promotion=False)
(w/'learned-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='core_route_records'},indent=2))
