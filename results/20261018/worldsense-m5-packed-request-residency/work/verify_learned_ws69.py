"""Independent post-terminal WS69 reconstruction of original oracles and actual leases."""
from pathlib import Path
import json,hashlib,datetime,sys
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-packed-request-residency';old=w.parent/'worldsense-request-weight-residency';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-packed-request-residency/source';load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
frozen=load(w/'learned-identities-before.json')
assert len(frozen['source'])==1721 and all(sha(s/k)==v for k,v in frozen['source'].items())and all(sha(Path(k))==v for k,v in frozen['external'].items())
for name,info in load(w/'learned-guard-adaptation-proof.json').items():
 original=(old/name).read_text();text=(w/name).read_text();assert hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']and hashlib.sha256(text.encode()).hexdigest()==info['current_sha256']
 for op in reversed(info['operations']):assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text.replace('worldsense-packed-request-residency','worldsense-request-weight-residency')==original
terminal=load(w/'learned-controller-terminal.json');assert terminal['success']and terminal['normal_restored']and not terminal['promotion']
journal=load(w/'learned-controller-journal.json');assert len(journal)==19 and all(v['returncode']==0 and v['ended_utc']>=v['started_utc']for v in journal)
assert all(load(w/'api-correctness-result.json').values())
corpus=load(w/'corpus-GPU-bits.json');t=corpus['totals']
assert t['packed_matrices']==201 and t['raw_small_matrices']==245 and t['HOST_raw_FP32_words_checked']==1724383232 and t['retained_primary_and_small_raw_bytes']==9850930176
assert t['original_bytes']==11850481664 and t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600
assert t['GPU_prefill_tile_FP32_words_checked']=={k:4200857600 for k in ['32x32','64x32','64x64']}
assert len(corpus['matrices'])==446
selection=load(w.parent/'worldsense-primary-blocks/matrix-selection.json')['matrices'];assert {(v['model'],v['tensor'],v['rows'],v['inner'])for v in selection}=={(v['model'],v['tensor'],v['rows'],v['inner'])for v in corpus['matrices']}
for row in corpus['matrices']:
 assert row['owner_accounting_restored'];count=row['rows']*row['inner']
 if row['packed']:assert row['GPU_original_FP32_words_checked']==row['GPU_vector_original_FP32_words_checked']==count and row['GPU_prefill_tile_FP32_words_checked']=={k:count for k in ['32x32','64x32','64x64']}
 else:assert row['HOST_raw_FP32_words_checked']==count
leased=load(w/'packed-lease-corpus.json');lt=leased['totals'];assert leased['status']=='all201_two_original_readers_exact_under_actual_packed_lease'
assert lt['matrices']==201 and lt['skipped_raw']==245 and lt['scalar_original_FP32_words']==lt['vector_original_FP32_words']==4200857600
assert len(leased['matrices'])==201 and {(v['model'],v['tensor'],v['rows'],v['inner'])for v in leased['matrices']}=={(v['model'],v['tensor'],v['rows'],v['inner'])for v in corpus['matrices']if v['packed']}
def inactive(q):
 assert not q['scope_open'] and not q['attached'] and not q['requested']
 assert q['active_bytes']==q['global_used_bytes']==q['set_allocation_count']==q['owner_count']==0
total_alloc=0
for index,row in enumerate(leased['matrices'],1):
 assert row['owner_accounting_restored']and row['serving_counters_unchanged']and row['scalar_original_FP32_words']==row['vector_original_FP32_words']==row['rows']*row['inner']
 q=row['active_lease'];e=row['after_cleanup'];assert q['calls']==q['admitted']==index and q['ends']==index-1 and q['fallback']==0
 assert q['last_count']==q['attempted_count']==q['set_allocation_count']==2 and q['owner_count']==1 and q['scope_open']and q['attached']and q['requested']
 assert q['active_bytes']==q['global_used_bytes']==q['last_bytes']==q['attempted_bytes']>0
 inactive(e);assert e['calls']==e['admitted']==e['ends']==index and e['fallback']==0;total_alloc+=q['last_bytes']
assert total_alloc==lt['sum_individual_actual_allocation_bytes']
assert load(w/'direct-upload-corpus.json')['totals']==dict(weights=546,FP32_words_checked=334609152)
resident=load(w/'resident-results.json');assert len(resident)==8 and {v['mode']for v in resident}=={'off','primary','on','compat56'}and all(v['returncode']==0 for v in resident)
private_steps=0
for row in resident:
 lines=(w/row['log']).read_text().splitlines();steps=[json.loads(line.split(' ',1)[1])for line in lines if line.startswith('private_kv_step ')]
 assert len(steps)==8 and all(v['exact_output_and_state']for v in steps)and any(v.startswith('private_kv_reference PASS ')for v in lines);private_steps+=len(steps)
for role in ['off','on','fault','compat56']:assert 'Four-input trained linear generic-resident parity/host-private isolation/inactive/malformed/shared-after-source-close/rebuild PASS '+role in(w/('trained-linear-'+role+'.log')).read_text()
vlm=load(w.parent/'worldsense-prefix-cache/state-guard.json')['records'][1];prefix={v['prompt_index']:v for v in load(w.parent/'worldsense-prefix-cache/prefix-model-guard.json')['records']}
records={role:load(w/('api-state-progress-'+role+'.json'))['records']for role in ['off','primary','on','compat56']}
for role,rows in records.items():
 assert len(rows)==5
 for i,row in enumerate(rows):
  ref=vlm if i<2 else prefix[[0,1,0][i-2]];c=row['call'];t=c['timings'];assert row['states']==ref['states']and c['generated_token_ids']==ref['generated_token_ids']
  baseline=records['off'][i]['call'];assert all(c[k]==baseline[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
  enabled=role!='off';compat=role=='compat56'
  for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:assert(cap in c['backend'])==enabled
  for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_full_isolated_library','cached_matrix_metadata','request_scope_weight_residency']:assert(cap in c['backend'])==compat
  assert('packed_request_weight_residency'in c['backend'])==(role=='on')
  assert t['lazy_default_rehydrated_bytes']==0 and(t['lazy_default_retired_bytes']>0)==enabled and(t['lazy_default_planned_slots']>0)==enabled
  p=t['primary_storage_after'];delta=t['primary_encoded_delta']
  if enabled and i<2:
   assert p['active_weights']==201 and p['original_bytes']==8401715200 and p['payload_bytes']+p['metadata_bytes']==6402163712 and p['metadata_bytes']==32819200
   assert all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])and delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']
  else:assert all(v==0 for v in delta.values())
  if role=='on':
   q=t['packed_request_weight_residency'];inactive(q);eligible=i<2;assert t['packed_request_eligible']==eligible and q['limit_bytes']==8589934592
   if eligible:
    assert q['calls']==q['admitted']==q['ends']==[1,3][i]and q['fallback']==q['coverage_fallbacks']==0 and q['reason']==1
    assert q['last_count']==q['attempted_count']==402 and q['last_bytes']==q['attempted_bytes']==p['allocated_bytes']
    assert q['coverage_checks']>0 and q['coverage_cache_hits']>0 and 33<q['last_coverage_graphs']<=256
   else:assert all(q[k]==0 for k in ['calls','ends','admitted','fallback','coverage_checks','coverage_cache_hits'])
  if compat:
   q=t['request_weight_residency'];assert q['calls']==q['ends']and q['active_bytes']==q['global_used_bytes']==0 and not q['scope_open']and q['limit_bytes']==4294967296
   if i<2:assert q['reason']==7 and q['admitted']==0 and q['fallback']==q['calls']
   else:assert q['reason']==1 and q['last_count']==197 and q['last_bytes']==3440902144 and q['coverage_fallbacks']==0
for role in ['off','on','compat56']:assert load(w/('vision-grid-'+role+'.json'))==load(w.parent/'worldsense-request-residency/vision-grid-off.json')
assert load(w/'request-session-guard-compat56.json')['all_cases_pass']
assert load(w/'request-full-session-guard-compat56.json')['real197masters_global4GiB_contention_close_readmission_reset']
layer=load(w/'packed-layer-session-guard.json');assert layer['all_cases_pass']and len(layer['records'])==10
for row in layer['records']:inactive(row['stats'])
full=load(w/'packed-full-session-guard.json');assert full['all_cases_pass']and len(full['records'])==8 and full['actual_owner_count']==201 and full['actual_allocation_count']==402 and full['logical_packed_bytes']==6402163712
inv=load(w/'packed-full-graph-inventory.json');buffers={v['id']:v for v in inv['buffers']};refs=[v for v in inv['references']if v['representation']in ['primary_bf16_payload','primary_bf16_metadata']];ids={v['buffer_id']for v in refs}
assert len(ids)==402 and sum(buffers[i]['allocated_bytes']for i in ids)==full['actual_allocation_bytes']and sum(buffers[i]['length_bytes']for i in ids)==6402163712
recovered=next(v for v in full['records']if v['case']=='same_model_after_image_errors_original32layers_states_tokens_exact');assert recovered['states']==vlm['states']and recovered['call']['generated_token_ids']==vlm['generated_token_ids'];inactive(recovered['call']['timings']['packed_request_weight_residency'])
assert load(w/'native-independent-audit.json')['native_tests_pass']==54
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_terminal_session=int(sys.argv[1]),observed_exit_code=0,source1721_and_all_explicit_inputs_unchanged=True,original_guard_adaptations_inverse_exact=True,original_five_full_GPU_readers_EACH_words=4200857600,both_original_GPU_readers_under_actual_lease_EACH_words=4200857600,raw_HOST_words=1724383232,direct_words=334609152,resident_private_steps=private_steps,four_API_modes20calls_original32VLM28text_states_tokens_vision_exact=True,additional_two_error_calls_and_same_instance_original_recovery=True,actual4B_owners=201,actual4B_allocations=402,actual4B_declared_allocation_bytes=full['actual_allocation_bytes'],old_raw_policy_and_new_packed_lifecycle_pass=True,no_speed_claim=True,promotion=False)
(w/'learned-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
