"""WS59 measured serving admission, not a numerical/metric gate replacement."""
def native_route(call,enabled,primary_control=False):
 import math
 try:
  packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta'];model=call['model_id'];remote=packed and model=='Qwen3-1.7B'
  primary_caps=['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']
  vector_caps=['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library']
  if model not in ['Qwen3.5-4B','Qwen3-1.7B']:return False
  if model=='Qwen3.5-4B'and call['image_preprocess_backend']!='native-cpp':return False
  if 'wide_checkpoint_storage'in call['backend']:return False
  for cap in primary_caps:
   if(cap in call['backend'])!=(packed and not remote):return False
  for cap in vector_caps:
   if(cap in call['backend'])!=(enabled and not remote):return False
  if('native_text_worker'in call['backend'])!=(packed and not remote):return False
  if t['lazy_default_rehydrated_bytes']!=0 or t['memory_after_decode']['task_info_status']!=0:return False
  if packed and not remote:
   if t['lazy_default_retired_bytes']<=0 or t['lazy_default_planned_slots']<=0:return False
  elif t['lazy_default_retired_bytes']!=0 or t['lazy_default_planned_slots']!=0:return False
  if not packed:
   return 'native_gateway'not in t and all(v==0 for v in s.values())and all(v==0 for v in delta.values())
  g=t['native_gateway'];parent=g['parent_backend'];pid=g['parent_pid'];memory=t['memory_after_decode'];parts=memory['process_samples'];child=g['child']
  if g['route']!=('native_lrpc_raw_text'if remote else'local')or'native_text_worker'not in parent:return False
  if any((cap in parent)!=True for cap in primary_caps)or any((cap in parent)!=enabled for cap in vector_caps):return False
  if 'native_text_worker_user_initiated_qos'not in parent:return False
  caller=g['parent_thread_qos_requested']
  if caller['get_status']!=0 or caller['thread_id_status']!=0 or caller['active_scopes']!=0 or caller['end_errors']!=0:return False
  if remote:
   q=call['native_worker_qos'];a=q['before'];b=q['after'];scope=q['scope']
   if q['mode']!='user_initiated'or not scope['api_start_accepted']or not scope['closed']or scope['api_end_status']!=0 or scope['target_class']!=25 or scope['relative_priority']!=0:return False
   if any(x['get_status']!=0 or x['thread_id_status']!=0 for x in [a,b,scope['before'],scope['during'],scope['after']]):return False
   if len({x['thread_id']for x in [a,b,scope['before'],scope['during'],scope['after']]})!=1:return False
   if a['requested_class']!=b['requested_class']or a['requested_class']!=scope['during']['requested_class']:return False
   if a['active_scopes']!=0 or b['active_scopes']!=0 or b['end_errors']!=0 or scope['during']['active_scopes']!=1:return False
   if b['starts']!=a['starts']+1 or b['ends']!=a['ends']+1 or b['start_attempts']!=a['start_attempts']+1:return False
   if b['starts']!=call['native_worker_call_index']or b['ends']!=call['native_worker_call_index']:return False
  if g['model_memory_after_decode']['task_info_status']!=0:return False
  if not all(math.isfinite(g[k])for k in ['gateway_started_monotonic','model_started_monotonic','first_token_monotonic','raw_model_ttft_seconds','raw_model_total_seconds']):return False
  if not(g['gateway_started_monotonic']<=g['model_started_monotonic']<=g['first_token_monotonic']<=g['gateway_started_monotonic']+call['total_seconds']+.0001):return False
  if abs(call['ttft_seconds']-(g['first_token_monotonic']-g['gateway_started_monotonic']))>1e-9:return False
  if min(g['raw_model_ttft_seconds'],g['raw_model_total_seconds'])<0 or g['raw_model_ttft_seconds']>call['ttft_seconds']+.0001 or g['raw_model_total_seconds']>call['total_seconds']+.0001:return False
  if not math.isfinite(call['total_seconds'])or call['total_seconds']<call['ttft_seconds']:return False
  expected_pids={pid}
  if child is not None:
   support=child['support'];expected_pids.add(support['pid'])
   if support['parent_pid']!=pid or support['pid']==pid or not support['watchdog_active']or support['watch_parent_pid']!=pid:return False
   if child['protocol']!='garnet-native-text-v1'or not child['birth_id']or not child['clock_epoch_metadata_enabled']:return False
   if support['library_path']!='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-thread-qos-off/libgarnet.dylib'or support['source_root']!='/Users/shawn/CantorAI/out/research/worldsense-thread-qos/source':return False
   if any(cap in child['backend']for cap in primary_caps+vector_caps+['native_text_worker']):return False
   if remote and(child['backend']!=call['backend']or child['models']!=[model]or child['infer_calls']!=call['native_worker_call_index']):return False
   if child['inference_qos_mode']!='user_initiated':return False
   if remote and child['last_inference_qos']!=call['native_worker_qos']:return False
  elif remote:return False
  if len(parts)!=len(expected_pids)or {p['pid']for p in parts}!=expected_pids or memory['cohort_process_count']!=len(parts)or g['process_count']!=len(parts):return False
  if parts[0]['pid']!=pid:return False
  for p in parts:
   if p['task_info_status']!=0 or p['metal_registry_id']!=memory['metal_registry_id']:return False
   if not math.isfinite(p['snapshot_monotonic'])or not(g['first_token_monotonic']<=p['snapshot_monotonic']<=g['gateway_started_monotonic']+call['total_seconds']+.0001):return False
   if p['pid']!=pid and p['birth_id']!=child['birth_id']:return False
  for key in ['metal_allocated_bytes','task_physical_footprint_bytes','task_resident_bytes','task_compressed_bytes','task_internal_bytes','task_virtual_bytes']:
   if any(not isinstance(p[key],int)or isinstance(p[key],bool)or p[key]<0 for p in parts)or memory[key]!=sum(p[key]for p in parts):return False
  if memory['metal_recommended_working_set_bytes']!=parts[0]['metal_recommended_working_set_bytes']:return False
  before=g['parent_primary_storage_before'];after=g['parent_primary_storage_after'];parent_delta=g['parent_primary_encoded_delta']
  for key in delta:
   if after[key]<before[key]or parent_delta[key]!=after[key]-before[key]:return False
  if remote:
   if before!=after or any(v!=0 for v in parent_delta.values())or any(v!=0 for v in s.values())or any(v!=0 for v in delta.values()):return False
  elif after!=s or parent_delta!=delta:return False
  active=after['active_weights']
  if model=='Qwen3.5-4B':
   if active!=201 or not all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded']):return False
   if delta['vector_decode_encoded']!=delta['decode_encoded']or delta['vector_head_encoded']!=delta['head_encoded']:return False
  elif active not in [0,201]:return False
  if active==0:return all(after[k]==0 for k in ['active_weights','original_bytes','payload_bytes','metadata_bytes','allocated_bytes'])
  return after['original_bytes']==8401715200 and after['payload_bytes']+after['metadata_bytes']==6402163712 and after['metadata_bytes']==32819200 and 0<after['allocated_bytes']<8401715200
 except (KeyError,ValueError,TypeError,OverflowError):return False
