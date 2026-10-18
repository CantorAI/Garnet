def native_route(call,enabled,primary_control=False):

 if call['model_id']not in ['Qwen3.5-4B','Qwen3-1.7B']:return False
 t0=call['timings']
 if ('primary_only_lazy_defaults'in call['backend'])!=(enabled or primary_control):return False
 if any(cap in call['backend']for cap in ['primary_decode_groups2','primary_decode_groups8']):return False
 if 'primary_storage_after'in t0:
  for key in ['lazy_default_retired_bytes','lazy_default_planned_slots','lazy_default_rehydrated_bytes']:
   if type(t0.get(key))is not int or t0[key]<0:return False
 if 'native_text_worker'in call['backend']or'native_gateway'in t0:return False
 for cap in ['request_scope_weight_residency','packed_request_weight_residency','primary_simd_metadata','sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','sole_cooperative_prefill']:
  if cap in call['backend']:return False
 if any(key in t0 for key in ['request_weight_residency','packed_request_weight_residency','packed_request_eligible','memory_after_packed_request_cleanup']):return False
 for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','primary_full_isolated_library']:
  if (cap in call['backend'])!=enabled:return False
 for cap in ['direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata']:
  if cap in call['backend']:return False
 if 'primary_core_isolated_library'in call['backend']:return False
 if 'primary_storage_after'in t0:
  before=t0.get('primary_storage_before');after=t0.get('primary_storage_after');encoded=t0.get('primary_encoded_delta')
  fields={'active_weights','original_bytes','payload_bytes','metadata_bytes','allocated_bytes','uploads','decode_encoded','prefill_encoded','head_encoded','vector_decode_encoded','vector_head_encoded','core_library_decode_encoded','core_library_prefill_encoded','core_library_head_encoded'}
  expected={'decode_encoded','prefill_encoded','head_encoded','vector_decode_encoded','vector_head_encoded'}
  if not isinstance(before,dict)or not isinstance(after,dict)or set(before)!=fields or set(after)!=fields:return False
  if not isinstance(encoded,dict)or set(encoded)!=expected:return False
  if any(type(v)is not int or v<0 for obj in [before,after,encoded]for v in obj.values()):return False
  if any(after[k]<before[k]or encoded[k]!=after[k]-before[k]for k in expected):return False
  for key,oldkey in [('core_library_decode_encoded','decode_encoded'),('core_library_prefill_encoded','prefill_encoded'),('core_library_head_encoded','head_encoded')]:
   if after[key]<before[key]:return False
   if after[key]-before[key]!=0:return False
   if before[key]!=0 or after[key]!=0:return False
   if call['model_id']=='Qwen3-1.7B'and after[key]!=before[key]:return False
 elif enabled or any(key in t0 for key in ['primary_storage_before','primary_encoded_delta']):return False
 if not enabled:
  return 'dense_prefix_cache'in call['backend']and not any(k in call['backend']for k in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','primary_vector_prefill','primary_prefill_isolated_library','lazy_resident_defaults','cached_matrix_metadata','wide_primary_storage','direct_float_weight_upload'])and not any(k in call['timings']for k in ['primary_storage_after','primary_encoded_delta','memory_after_decode'])
 packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:
  if (cap in call['backend'])!=packed:return False
 for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library']:
  if cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','primary_full_isolated_library']:
   if (cap in call['backend'])!=enabled:return False
  elif cap in call['backend']:return False
 if 'wide_checkpoint_storage'in call['backend']:return False
 if t['lazy_default_rehydrated_bytes']!=0 or t['memory_after_decode']['task_info_status']!=0:return False
 if packed and call['model_id']=='Qwen3.5-4B':
  if t['lazy_default_retired_bytes']<=0 or t['lazy_default_planned_slots']<=0:return False
 elif t['lazy_default_retired_bytes']!=0 or t['lazy_default_planned_slots']!=0:return False
 if not packed:return all(v==0 for v in s.values())and all(v==0 for v in delta.values())
 active=s['active_weights'];model=call['model_id']
 if model=='Qwen3.5-4B':
  if active!=201:return False
  if not all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded']):return False
  if delta['vector_decode_encoded']!=delta['decode_encoded']or delta['vector_head_encoded']!=delta['head_encoded']:return False
 elif model=='Qwen3-1.7B':
  if active not in [0,201]or any(v!=0 for v in delta.values()):return False
 else:return False
 if active==0:return all(s[k]==0 for k in ['active_weights','original_bytes','payload_bytes','metadata_bytes','allocated_bytes'])
 return s['original_bytes']==8401715200 and s['payload_bytes']+s['metadata_bytes']==6402163712 and s['metadata_bytes']==32819200 and 0<s['allocated_bytes']<8401715200
