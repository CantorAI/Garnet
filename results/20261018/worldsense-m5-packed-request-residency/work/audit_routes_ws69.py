"""Offline mutated copies of actual calls; no new inference or timing samples."""
from pathlib import Path
import ast,json,copy,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-packed-request-residency';load=lambda p:json.loads(p.read_text());results={}
for rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
 fn=next(v for v in ast.parse((w/rel).read_text()).body if isinstance(v,ast.FunctionDef)and v.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'native admission','exec'),ns);route=ns['native_route'];positive=negative=legacy=0
 for mode in ['off','primary','on']:
  if rel=='production-speed/analyze.py'and mode!='on':continue
  for row in load(w/('api-state-progress-'+mode+'.json'))['records']:
   c=row['call'];enabled=mode=='on';primary=mode=='primary';assert route(c,enabled,primary),(rel,mode,c['model_id']);positive+=1
   def reject(value):
    global negative
    assert not route(value,enabled,primary),(rel,mode,c['model_id'],'malformed copy accepted');negative+=1
   for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults','primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library','request_scope_weight_residency','packed_request_weight_residency','sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','sole_cooperative_prefill','primary_simd_metadata','native_text_worker']:
    bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace(cap,'')if cap in bad['backend']else bad['backend']+' '+cap;reject(bad)
   for field,value in [('model_id','unknown')]:bad=copy.deepcopy(c);bad[field]=value;reject(bad)
   bad=copy.deepcopy(c);bad['timings']['native_gateway']={};reject(bad)
   bad=copy.deepcopy(c);bad['timings']['request_weight_residency']={};reject(bad)
   bad=copy.deepcopy(c);bad['timings']['memory_after_decode']['task_info_status']=1;reject(bad)
   for field in ['lazy_default_retired_bytes','lazy_default_planned_slots','lazy_default_rehydrated_bytes']:
    bad=copy.deepcopy(c);bad['timings'][field]=0 if mode!='off'and field!='lazy_default_rehydrated_bytes'else 1;reject(bad)
   if enabled:
    for field in ['packed_request_eligible','packed_request_weight_residency','memory_after_packed_request_cleanup']:
     bad=copy.deepcopy(c);bad['timings'].pop(field);reject(bad)
    bad=copy.deepcopy(c);bad['timings']['packed_request_eligible']=not bad['timings']['packed_request_eligible'];reject(bad)
    for key in c['timings']['packed_request_weight_residency']:
     bad=copy.deepcopy(c);bad['timings']['packed_request_weight_residency'].pop(key);reject(bad)
     original=c['timings']['packed_request_weight_residency'][key]
     for value in [None,[],{},'0',0.0]if type(original)is bool else [None,[],{},'0',True,False,-1,float('nan')]:
      bad=copy.deepcopy(c);bad['timings']['packed_request_weight_residency'][key]=value;reject(bad)
    changes={'enabled':False,'scope_open':True,'attached':True,'requested':True,'active_bytes':1,'global_used_bytes':1,'set_allocation_count':1,'owner_count':1,'limit_bytes':131072}
    if c['model_id']=='Qwen3.5-4B':changes.update(calls=0,admitted=0,ends=0,fallback=1,reason=8,last_bytes=6402163711,attempted_bytes=0,last_count=401,attempted_count=401,coverage_cache_hits=0,coverage_checks=0,coverage_fallbacks=1,last_coverage_graphs=257)
    else:changes.update({key:1 for key in c['timings']['packed_request_weight_residency']if key not in ['enabled','scope_open','attached','requested','limit_bytes']})
    for key,value in changes.items():bad=copy.deepcopy(c);bad['timings']['packed_request_weight_residency'][key]=value;reject(bad)
    for field in ['task_info_status','metal_allocated_bytes','task_physical_footprint_bytes','task_resident_bytes','task_compressed_bytes','task_internal_bytes','task_virtual_bytes']:
     for value in [None,False,-1,float('nan')]:bad=copy.deepcopy(c);bad['timings']['memory_after_packed_request_cleanup'][field]=value;reject(bad)
   else:
    for key in ['packed_request_eligible','packed_request_weight_residency','memory_after_packed_request_cleanup']:bad=copy.deepcopy(c);bad['timings'][key]={};reject(bad)
   if c['model_id']=='Qwen3-1.7B'and mode!='off':
    vlm=load(w/('api-state-progress-'+mode+'.json'))['records'][0]['call'];both=copy.deepcopy(c);both['timings']['primary_storage_after']=copy.deepcopy(vlm['timings']['primary_storage_after']);assert route(both,enabled,primary)
    bad=copy.deepcopy(both);bad['timings']['primary_storage_after']['active_weights']=398;reject(bad)
    bad=copy.deepcopy(both);bad['timings']['primary_encoded_delta']['decode_encoded']=1;reject(bad)
 if rel=='production-speed/analyze.py':
  for item in load(b/'work/worldsense-isolated-primary/production-speed/control-1.json')['items']:
   for c in item['result']['model_response'].values():
    if not isinstance(c,dict)or c.get('status')!='ok':continue
    assert route(c,False);legacy+=1
    for key in ['packed_request_eligible','packed_request_weight_residency','memory_after_packed_request_cleanup','request_weight_residency','native_gateway']:
     bad=copy.deepcopy(c);bad['timings'][key]={};assert not route(bad,False);negative+=1
    for cap in ['packed_request_weight_residency','request_scope_weight_residency','primary_simd_metadata','sole_interleaved_mlp']:
     bad=copy.deepcopy(c);bad['backend']+=' '+cap;assert not route(bad,False);negative+=1
  assert legacy==10
 results[rel]=dict(actual_full_API_calls=positive,malformed_copies_rejected=negative,actual_legacy_WS10_calls=legacy)
p=w/'acceptance-gate-audit.json';d=load(p);d.update(full_route_mutation_audit_pending=False,full_route_mutation_audit_pass=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_route_audit=results,scope='Offline actual full API calls and corrupt copies. All five original numeric/golden/metric gates unchanged; no inference/speed evidence.');p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(results,indent=2))
