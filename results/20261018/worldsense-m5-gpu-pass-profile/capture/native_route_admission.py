def native_route(call,enabled,primary_control=False):
 packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:
  if (cap in call['backend'])!=packed:return False
 for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library']:
  if (cap in call['backend'])!=enabled:return False
 if 'wide_checkpoint_storage'in call['backend']:return False
 if t['lazy_default_rehydrated_bytes']!=0 or t['memory_after_decode']['task_info_status']!=0:return False
 if packed:
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
