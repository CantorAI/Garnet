"""Offline actual native calls and corrupted copies, no inference or timing samples."""
from pathlib import Path
import ast,json,copy,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-minimal-full-prefill';load=lambda p:json.loads(p.read_text());results={}
for rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
 fn=next(v for v in ast.parse((w/rel).read_text()).body if isinstance(v,ast.FunctionDef)and v.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'native admission','exec'),ns);route=ns['native_route'];positive=negative=legacy=0
 for mode in ['off','primary','on']:
  if rel=='production-speed/analyze.py'and mode!='on':continue
  for row in load(w/('api-state-progress-'+mode+'.json'))['records']:
   c=row['call'];enabled=mode=='on';primary=mode=='primary';assert route(c,enabled,primary),(rel,mode,c['model_id']);positive+=1
   def reject(value):
    global negative
    assert not route(value,enabled,primary),(rel,mode,c['model_id'],'malformed copy accepted');negative+=1
   for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults','primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library','primary_core_isolated_library','request_scope_weight_residency','packed_request_weight_residency','sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','sole_cooperative_prefill','primary_simd_metadata','native_text_worker']:
    bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace(cap,'')if cap in bad['backend']else bad['backend']+' '+cap;reject(bad)
   bad=copy.deepcopy(c);bad['model_id']='unknown';reject(bad)
   for field in ['native_gateway','request_weight_residency','packed_request_weight_residency','packed_request_eligible','memory_after_packed_request_cleanup']:bad=copy.deepcopy(c);bad['timings'][field]={};reject(bad)
   bad=copy.deepcopy(c);bad['timings']['memory_after_decode']['task_info_status']=1;reject(bad)
   for field in ['lazy_default_retired_bytes','lazy_default_planned_slots','lazy_default_rehydrated_bytes']:
    bad=copy.deepcopy(c);bad['timings'][field]=0 if mode!='off'and field!='lazy_default_rehydrated_bytes'else 1;reject(bad)
   for group in ['primary_storage_before','primary_storage_after','primary_encoded_delta']:
    bad=copy.deepcopy(c);bad['timings'].pop(group);reject(bad)
    for invalid in [None,[],1,False]:bad=copy.deepcopy(c);bad['timings'][group]=invalid;reject(bad)
    for key in c['timings'][group]:
     bad=copy.deepcopy(c);bad['timings'][group].pop(key);reject(bad)
     for invalid in [None,True,False,-1,0.0,'0',float('nan')]:bad=copy.deepcopy(c);bad['timings'][group][key]=invalid;reject(bad)
   for group in ['primary_storage_before','primary_storage_after']:
    for key in ['core_library_decode_encoded','core_library_prefill_encoded','core_library_head_encoded']:
     bad=copy.deepcopy(c);bad['timings'][group][key]+=1;reject(bad)
   for key in c['timings']['primary_encoded_delta']:bad=copy.deepcopy(c);bad['timings']['primary_encoded_delta'][key]+=1;reject(bad)
   if c['model_id']=='Qwen3-1.7B'and mode!='off':
    vlm=load(w/('api-state-progress-'+mode+'.json'))['records'][0]['call'];both=copy.deepcopy(c)
    # Global VLM weight ownership may remain while the companion has no packed work.
    for key in ['active_weights','original_bytes','payload_bytes','metadata_bytes','allocated_bytes']:both['timings']['primary_storage_after'][key]=vlm['timings']['primary_storage_after'][key]
    assert route(both,enabled,primary)
    bad=copy.deepcopy(both);bad['timings']['primary_storage_after']['active_weights']=398;reject(bad)
 if rel=='production-speed/analyze.py':
  for item in load(b/'work/worldsense-isolated-primary/production-speed/control-1.json')['items']:
   for c in item['result']['model_response'].values():
    if not isinstance(c,dict)or c.get('status')!='ok':continue
    assert route(c,False);legacy+=1
    for field in ['primary_storage_before','primary_storage_after','primary_encoded_delta','packed_request_weight_residency','request_weight_residency','native_gateway']:
     bad=copy.deepcopy(c);bad['timings'][field]={};assert not route(bad,False);negative+=1
    for cap in ['primary_core_isolated_library','packed_request_weight_residency','request_scope_weight_residency','primary_simd_metadata','primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','primary_full_isolated_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata']:
     bad=copy.deepcopy(c);bad['backend']+=' '+cap;assert not route(bad,False);negative+=1
  assert legacy==10
 results[rel]=dict(actual_full_API_calls=positive,malformed_copies_rejected=negative,actual_legacy_WS10_calls=legacy)
p=w/'acceptance-gate-audit.json';d=load(p);d.update(full_route_mutation_audit_pending=False,full_route_mutation_audit_pass=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_route_audit=results,scope='Offline copied actual calls. Strict14field snapshots/5encoded deltas/core provenance and legacy compatibility; original five quantitative/golden gates unchanged. No new inference/speed samples.');p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(results,indent=2))
