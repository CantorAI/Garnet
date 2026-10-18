"""Independent post-terminal original learned oracles and complete reader coverage."""
from pathlib import Path
import hashlib,json,datetime,sys
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-primary-simd-metadata';old=w.parent/'worldsense-cooperative-prefill';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-simd-metadata/source';load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
frozen=load(w/'learned-identities-before.json');assert len(frozen['source'])==1726 and all(sha(s/k)==v for k,v in frozen['source'].items())and all(sha(Path(k))==v for k,v in frozen['external'].items())
for name,info in load(w/'learned-guard-adaptation-proof.json').items():
 original=(old/name).read_text();text=(w/name).read_text();assert hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']and hashlib.sha256(text.encode()).hexdigest()==info['current_sha256']
 for op in reversed(info['operations']):assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text.replace('worldsense-primary-simd-metadata','worldsense-cooperative-prefill')==original
assert all(load(w/'api-correctness-result.json').values())
corpus=load(w/'corpus-GPU-bits.json');t=corpus['totals'];assert t['packed_matrices']==201 and t['raw_small_matrices']==245 and t['HOST_raw_FP32_words_checked']==1724383232 and t['retained_primary_and_small_raw_bytes']==9850930176
assert t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600 and all(v==4200857600 for v in t['GPU_prefill_tile_FP32_words_checked'].values())and set(t['GPU_prefill_tile_FP32_words_checked'])=={'32x32','64x64','64x32'}
sole=[v for v in corpus['matrices']if v['GPU_sole_original_FP32_words_checked']];assert len(sole)==84 and sum(v['GPU_sole_original_FP32_words_checked']for v in sole)==1056964608
assert t['GPU_sole_prefill_tile_FP32_words_checked']=={'32x32':1056964608,'64x64':1056964608,'64x32':1056964608}
for row in sole:assert row['model']=='Qwen3-1.7B'and row['GPU_sole_original_FP32_words_checked']==row['rows']*row['inner']and row['GPU_sole_prefill_tile_FP32_words_checked']=={key:row['rows']*row['inner']for key in ['32x32','64x64','64x32']}
new=load(w/'simd-metadata-corpus.json');assert new['status']=='both_exact_serving_helpers_all201_original_words_exact'and new['totals']==dict(matrices=201,skipped_raw=245,decode_helper_original_FP32_words_checked=4200857600,prefill_helper_original_FP32_words_checked=4200857600)
expected={(row['model'],row['tensor']):(row['rows'],row['inner'])for row in corpus['matrices']if row['packed']};assert len(new['matrices'])==201 and {(row['model'],row['tensor']):(row['rows'],row['inner'])for row in new['matrices']}==expected
for row in new['matrices']:assert row['owner_accounting_restored']and row['serving_counters_unchanged']and row['decode_helper_original_FP32_words_checked']==row['prefill_helper_original_FP32_words_checked']==row['rows']*row['inner']
assert load(w/'direct-upload-corpus.json')['totals']==dict(weights=546,FP32_words_checked=334609152)
paired=load(w/'dual-pair-corpus.json');assert paired['totals']==dict(pairs=28,matrices=56,original_FP32_words_checked=704643072)
names={row['tensor']for row in load(w.parent/'worldsense-primary-blocks/matrix-selection.json')['matrices']if row['model']=='Qwen3-1.7B'and row['tensor'].endswith(('.mlp.gate_proj.weight','.mlp.up_proj.weight'))};assert names=={row[key]for row in paired['pairs']for key in ['gate','up']}
assert all(row['same_decode_vector_helper']and row['owner_accounting_restored']and row['serving_counters_unchanged']and row['original_FP32_words_checked']==25165824 for row in paired['pairs'])
resident=load(w/'resident-results.json');assert len(resident)==10 and {row['mode']for row in resident}=={'off','primary','on','compat67','profile-compat67'}and all(row['returncode']==0 for row in resident)
for row in resident:
 line=next(line for line in(w/row['log']).read_text().splitlines()if line.startswith('dual_projection_generic_resident_routes '));d=json.loads(line.split(' ',1)[1]);enabled=row['mode']=='compat67';assert d['enabled']==enabled and len(d['checks'])==34 and all(v==dict(encoded_pairs=int(enabled),represented_projections=2*int(enabled))for v in d['checks'])
vlm=load(w.parent/'worldsense-prefix-cache/state-guard.json')['records'][1];prefix={v['prompt_index']:v for v in load(w.parent/'worldsense-prefix-cache/prefix-model-guard.json')['records']};records={role:load(w/('api-state-progress-'+role+'.json'))['records']for role in ['off','primary','on','compat67']}
for role,rows in records.items():
 assert len(rows)==5
 for i,row in enumerate(rows):
  ref=vlm if i<2 else prefix[[0,1,0][i-2]];c=row['call'];assert row['states']==ref['states']and c['generated_token_ids']==ref['generated_token_ids'];baseline=records['off'][i]['call'];assert all(c[key]==baseline[key]for key in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
  t=c['timings'];enabled=role!='off';full=role=='compat67';assert ('primary_simd_metadata'in c['backend'])==(role=='on')
  for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:assert(cap in c['backend'])==enabled
  for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_full_isolated_library','cached_matrix_metadata','sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','sole_cooperative_prefill']:assert(cap in c['backend'])==full
  assert t['lazy_default_rehydrated_bytes']==0 and(t['lazy_default_retired_bytes']>0)==enabled and(t['lazy_default_planned_slots']>0)==enabled
  stats=t['primary_storage_after'];delta=t['primary_encoded_delta'];sole=t['sole_storage_after'];sd=t['sole_encoded_delta'];dd=t['dual_projection_delta'];db=t['dual_projection_before'];da=t['dual_projection_after'];assert set(dd)==set(db)==set(da)=={'encoded_pairs','represented_projections'}and dd=={k:da[k]-db[k]for k in dd}and all(v['represented_projections']==2*v['encoded_pairs']for v in [dd,db,da])
  if enabled and i<2:assert stats['active_weights']==201 and stats['original_bytes']==8401715200 and stats['payload_bytes']+stats['metadata_bytes']==6402163712 and stats['metadata_bytes']==32819200 and all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])and delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']
  else:assert all(v==0 for v in delta.values())
  if full and i>=2:assert sole['active_weights']==84 and sole['allocated_bytes']==2113929216 and sole['scratch_bytes']==0 and sole['restore_encoded']==0 and sd['decode_encoded']>0 and dd['encoded_pairs']>=28*(c['generated_tokens']-1)>0
  else:assert all(v==0 for v in sole.values())and all(v==0 for v in sd.values())and all(v==0 for v in dd.values())
for role in ['off','on','compat67']:assert load(w/('vision-grid-'+role+'.json'))==load(w.parent/'worldsense-request-residency/vision-grid-off.json')
assert load(w/'native-independent-audit.json')['native_tests_pass']==50
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_terminal_session=int(sys.argv[1]),observed_exit_code=0,source1726_and_all_explicit_frozen_inputs_unchanged=True,all_original_guard_adaptations_inverse_exact=True,all_original_five_packed_GPU_readers_each_FP32_words=4200857600,both_new_GPU_helpers_EACH_all201_original_FP32_words=4200857600,all245raw_HOST_FP32_words=1724383232,all84sole_scalar_and_each_three_tile_words=1056964608,all56paired_helper_words=704643072,direct_words=334609152,real_resident_profile_route_checks=340,full_four_actual_API_modes20calls_original32VLM28text_states_tokens_vision_exact=True,defaultOFF_compat67_exact=True,no_speed_claim=True,promotion=False)
(w/'learned-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
