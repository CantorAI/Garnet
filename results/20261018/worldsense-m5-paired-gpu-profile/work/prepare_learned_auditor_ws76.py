"""Prospectively retain WS74 independent numerical audit and add trace proof."""
from pathlib import Path
import ast,hashlib,json
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');opt=b/'work/worldsense-opt';main=b/'work/worldsense-paired-gpu-profile'
old=(opt/'verify_learned_ws74.py').read_text();t=old.replace('worldsense-minimal-selective-defaults','worldsense-paired-gpu-profile-current');ops=[]
def change(a,z,count=1):
 global t
 assert t.count(a)==count,(a[:100],t.count(a));t=t.replace(a,z);ops.append(dict(before=a,after=z,count=count))
change("old=w.parent/'worldsense-minimal-full-prefill'","old=w.parent/'worldsense-minimal-selective-defaults'")
change("text.replace('worldsense-paired-gpu-profile-current','worldsense-minimal-full-prefill')", "text.replace('worldsense-paired-gpu-profile-current','worldsense-minimal-selective-defaults')")
change("len(frozen['source'])==1721","len(frozen['source'])==1723")
change('len(journal)==16','len(journal)==25')
change("for role in ['off','primary','on','compat56']}","for role in ['off','primary','on','quiet','compat56']}")
change("vector=role in ['on','compat56']","vector=role in ['on','quiet','compat56']")
change("selective=role in ['primary','on']","selective=role in ['primary','on','quiet']")
change("assert load(w/'native-independent-audit.json')['native_tests_pass']==54","assert load(w.parent/'worldsense-paired-gpu-profile/compatibility-native-independent-audit.json')['combined_native_CTest_count']==79")
change('source1721_and_all_explicit_inputs_unchanged','source1723_and_all_explicit_inputs_unchanged')
change('four_API_modes20calls_original32VLM28text_states_tokens_vision_exact','five_current_API_modes25calls_original32VLM28text_states_tokens_vision_exact')
extra='''
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
(main/'learned-counter-command-proof.json').write_text(json.dumps(dict(proofs=trace_proofs,all_expected_selected_commands_and_complete_intervals_joined=True,all_aligned_ordered_operations_equal_after_declared_raw_primary_label_normalization=True,no_performance_claim=True),indent=2)+'\\n')
'''
anchor='result=dict(utc=';assert t.count(anchor)==1;t=t.replace(anchor,extra+'\n'+anchor);ops.append(dict(before=anchor,after=extra+'\n'+anchor,count=1))
change('promotion=False)','legacy_three_API_modes15calls_and_three_vision_modes_exact=True,both_actual_learned_counter_errors_scope_cleanup_and28state_recovery_exact=True,all_selected_counter_commands_and_aligned_ordered_operation_sequences_audited=True,no_actual_WorldSense_capture=True,promotion=False)')
ast.parse(t);inverse=t
for op in reversed(ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
assert inverse.replace('worldsense-paired-gpu-profile-current','worldsense-minimal-selective-defaults')==old
target=opt/'verify_learned_ws76.py';assert not target.exists();target.write_text(t)
(main/'independent-learned-auditor-adaptation-proof.json').write_text(json.dumps(dict(seed=str(opt/'verify_learned_ws74.py'),seed_sha256=hashlib.sha256(old.encode()).hexdigest(),current=str(target),current_sha256=hashlib.sha256(t.encode()).hexdigest(),operations=ops,original_all_quantitative_checks_inverse_exact=True,no_learned_dispatch_yet=True),indent=2)+'\n')
print('Original74 independent learned audit retained; both legacy/current counter/state/vision/fault/command proofs added before learned dispatch')
