"""Independent original learned oracles, actual core words/counters and input closure."""
from pathlib import Path
import json,hashlib,datetime,sys
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-primary-decode-groups2';old=w.parent/'worldsense-minimal-full-prefill';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-decode-groups2/source';load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
frozen=load(w/'learned-identities-before.json');assert len(frozen['source'])==1723 and all(sha(s/k)==v for k,v in frozen['source'].items())and all(sha(Path(k))==v for k,v in frozen['external'].items())
for name,info in load(w/'learned-guard-adaptation-proof.json').items():
 original=(old/name).read_text();text=(w/name).read_text();assert hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']and hashlib.sha256(text.encode()).hexdigest()==info['current_sha256']
 for op in reversed(info['operations']):assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text.replace('worldsense-primary-decode-groups2','worldsense-minimal-full-prefill')==original
terminal=load(w/'learned-controller-terminal.json');assert terminal['success']and terminal['normal_restored']and not terminal['promotion'];journal=load(w/'learned-controller-journal.json');assert len(journal)==16 and all(v['returncode']==0 and v['ended_utc']>=v['started_utc']for v in journal)
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
vlm=load(w.parent/'worldsense-prefix-cache/state-guard.json')['records'][1];prefix={v['prompt_index']:v for v in load(w.parent/'worldsense-prefix-cache/prefix-model-guard.json')['records']};records={role:load(w/('api-state-progress-'+role+'.json'))['records']for role in ['off','primary','on','compat56']};core_routes=[]
for role,rows in records.items():
 assert len(rows)==5
 for index,row in enumerate(rows):
  reference=vlm if index<2 else prefix[[0,1,0][index-2]];call=row['call'];timings=call['timings'];assert row['states']==reference['states']and call['generated_token_ids']==reference['generated_token_ids'];baseline=records['off'][index]['call'];assert all(call[k]==baseline[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
  enabled=role!='off';compat=role=='compat56';isolated=False;vector=role in ['on','compat56']
  for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:assert(cap in call['backend'])==enabled
  for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_full_isolated_library']:assert(cap in call['backend'])==vector
  for cap in ['cached_matrix_metadata','request_scope_weight_residency']:assert(cap in call['backend'])==compat
  assert('primary_decode_groups2'in call['backend'])==(role=='on')
  for key,oldkey in [('decode_groups2_encoded','decode_encoded'),('head_groups2_encoded','head_encoded')]:
   before_group=timings['primary_storage_before'][key];after_group=timings['primary_storage_after'][key]
   assert type(before_group)is int and type(after_group)is int and 0<=before_group<=after_group
   assert after_group-before_group==(timings['primary_encoded_delta'][oldkey]if role=='on'else 0)
   if role!='on':assert before_group==after_group==0
   if index>=2:assert before_group==after_group
  assert('primary_core_isolated_library'in call['backend'])==isolated
  assert timings['lazy_default_rehydrated_bytes']==0 and(timings['lazy_default_retired_bytes']>0)==enabled and(timings['lazy_default_planned_slots']>0)==enabled
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
assert load(w/'native-independent-audit.json')['native_tests_pass']==60
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_terminal_session=int(sys.argv[1]),observed_exit_code=0,source1723_and_all_explicit_inputs_unchanged=True,original_guard_adaptations_inverse_exact=True,original_five_full_GPU_readers_EACH_words=4200857600,all_five_readers_in_actual_FULL_candidate=True,raw_HOST_words=1724383232,direct_words=334609152,resident_private_steps=private_steps,four_API_modes20calls_original32VLM28text_states_tokens_vision_exact=True,actual_core_deltas_zero_all_modes=True,actual_groups2_deltas_match_original_primary_and_zero_text=True,old_raw_compatibility_lifecycles_pass=True,core_route_records=core_routes,no_speed_claim=True,promotion=False)
(w/'learned-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='core_route_records'},indent=2))
