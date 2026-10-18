"""Actual native text-call provenance/clock/logical-counter admission."""
from pathlib import Path
import os,math
def validated_call(call,mode):
 assert call['status']=='ok'and call['model_id']=='Qwen3-1.7B'
 root=Path('/Users/shawn/CantorAI');source=root/'out/research/worldsense-validated-text-worker/source';build=root/('out/build/garnet-metal-worldsense-validated-text-worker-'+mode);old=root/'out/research/worldsense-prefix-cache/source'
 t=call['timings'];g=t['native_gateway'];child=g['child'];support=child['support'];model=child['model_backend']
 assert 'native_text_worker_validated_ws10'in g['parent_backend']and g['route']=='native_lrpc_raw_text'and g['parent_pid']==os.getpid()
 assert Path(support['library_path']).resolve()==build/'libgarnet_worker_host.dylib'and Path(support['source_root']).resolve()==source
 assert support['pid']!=os.getpid()and support['parent_pid']==os.getpid()and support['watch_parent_pid']==os.getpid()and support['watchdog_active']
 assert model['library_path']==str(root/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib')and model['library_sha256']=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'and model['loaded_garnet_images']==1 and model['package_abi']>0
 assert Path(model['expected_original_source_root']).resolve()==old
 assert Path(child['decoder_source']).resolve()==old/'xModel/qwen/metal_dense/decoder.py'
 assert Path(child['serving_source']).resolve()==source/'runtime/validated_ws10/metal_serving.py'and Path(t['serving_source']).resolve()==source/'runtime/validated_ws10/metal_serving.py'
 assert child['raw_counter_policy']=='pinned_ws10_structural_absence'and child['raw_counter_origin']==t['counter_origin']and 'Structural zeros'in t['counter_origin']
 assert child['infer_calls']==call['native_worker_call_index']and child['models']==[call['model_id']]and child['birth_id']and child['clock_epoch_metadata_enabled']
 for key in ['primary_storage_before','primary_storage_after','primary_encoded_delta']:
  assert all(isinstance(v,int)and not isinstance(v,bool)and v==0 for v in t[key].values())
 for key in ['lazy_default_planned_slots','lazy_default_retired_bytes','lazy_default_rehydrated_bytes']:assert isinstance(t[key],int)and not isinstance(t[key],bool)and t[key]==0
 assert g['parent_primary_storage_before']==g['parent_primary_storage_after']and all(v==0 for v in g['parent_primary_encoded_delta'].values())
 assert all(math.isfinite(g[k])for k in ['gateway_started_monotonic','model_started_monotonic','first_token_monotonic','raw_model_ttft_seconds','raw_model_total_seconds'])
 assert g['gateway_started_monotonic']<=g['model_started_monotonic']<=g['first_token_monotonic']<=g['gateway_started_monotonic']+call['total_seconds']+.0001
 assert abs(g['raw_model_ttft_seconds']-(g['first_token_monotonic']-g['model_started_monotonic']))<1e-9
 assert abs(call['ttft_seconds']-(g['first_token_monotonic']-g['gateway_started_monotonic']))<1e-9
 memory=t['memory_after_decode'];parts=memory['process_samples'];assert memory['cohort_process_count']==2 and len(parts)==2 and {p['pid']for p in parts}=={os.getpid(),support['pid']}
 for key in ['metal_allocated_bytes','task_physical_footprint_bytes','task_resident_bytes','task_compressed_bytes','task_internal_bytes','task_virtual_bytes']:
  assert all(isinstance(p[key],int)and not isinstance(p[key],bool)and p[key]>=0 for p in parts)and memory[key]==sum(p[key]for p in parts)
 assert all(p['task_info_status']==0 and p['metal_registry_id']==memory['metal_registry_id']for p in parts)
 assert memory['metal_recommended_working_set_bytes']==parts[0]['metal_recommended_working_set_bytes']
