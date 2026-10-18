"""Offline native admission audit using actual retained calls and malformed copies."""
from pathlib import Path
import ast,json,copy,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-direct-interleaved-prefill';results={}
for rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
 t=(w/rel).read_text();fn=next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)and n.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'native admission','exec'),ns);route=ns['native_route'];positive=negative=legacy=0
 for mode in ['off','primary','on']:
  if rel=='production-speed/analyze.py'and mode!='on':continue
  records=json.loads((w/('api-state-progress-'+mode+'.json')).read_text())['records'];assert len(records)==5
  for row in records:
   c=row['call'];enabled=mode=='on';primary=mode=='primary';assert route(c,enabled,primary),(rel,mode,c['model_id']);positive+=1
   def reject(value):
    global negative
    assert not route(value,enabled,primary),(rel,mode,'malformed native copy accepted');negative+=1
   for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults','primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library','sole_interleaved_mlp','sole_direct_interleaved_prefill']:
    bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace(cap,'')if cap in bad['backend']else bad['backend']+' '+cap;reject(bad)
   for group in ['sole_storage_before','sole_storage_after','sole_encoded_delta']:
    bad=copy.deepcopy(c);bad['timings'].pop(group);reject(bad)
    for invalid in [None,[],1,False]:bad=copy.deepcopy(c);bad['timings'][group]=invalid;reject(bad)
    for key in c['timings'][group]:
     for invalid in [None,-1,True,False,'1',float('nan')]:bad=copy.deepcopy(c);bad['timings'][group][key]=invalid;reject(bad)
     bad=copy.deepcopy(c);bad['timings'][group].pop(key);reject(bad)
   for field in ['decode_encoded','restore_encoded','prefill_encoded']:
    bad=copy.deepcopy(c);bad['timings']['sole_encoded_delta'][field]+=1;reject(bad)
   bad=copy.deepcopy(c);bad['model_id']='unknown';reject(bad)
   bad=copy.deepcopy(c);bad['backend']+=' native_text_worker';reject(bad)
   bad=copy.deepcopy(c);bad['timings']['native_gateway']={};reject(bad)
   bad=copy.deepcopy(c);bad['timings']['memory_after_decode']['task_info_status']=1;reject(bad)
   bad=copy.deepcopy(c);bad['timings']['lazy_default_rehydrated_bytes']+=1;reject(bad)
   if mode=='on':
    bad=copy.deepcopy(c);bad['timings']['sole_storage_after']['allocated_bytes']+=1;reject(bad)
    bad=copy.deepcopy(c);bad['timings']['sole_storage_after']['scratch_bytes']+=1;reject(bad)
    if c['model_id']=='Qwen3-1.7B':
     for field in ['active_weights','allocated_bytes']:bad=copy.deepcopy(c);bad['timings']['sole_storage_after'][field]=0;reject(bad)
 if rel=='production-speed/analyze.py':
  data=json.loads((w.parent/'worldsense-isolated-primary/production-speed/control-1.json').read_text())
  for item in data['items']:
   for c in item['result']['model_response'].values():
    if not isinstance(c,dict)or c.get('status')!='ok':continue
    assert route(c,False);legacy+=1
    for field in ['sole_storage_before','sole_storage_after','sole_encoded_delta','native_gateway']:
     bad=copy.deepcopy(c);bad['timings'][field]={};assert not route(bad,False);negative+=1
    bad=copy.deepcopy(c);bad['backend']+=' sole_interleaved_mlp';assert not route(bad,False);negative+=1
  assert legacy==10
 results[rel]=dict(actual_full_API_calls=positive,malformed_copied_fixtures_rejected=negative,actual_legacy_WS10_calls=legacy)
p=w/'acceptance-gate-audit.json';d=json.loads(p.read_text());d.update(full_route_mutation_audit_pending=False,full_route_mutation_audit_pass=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_route_audit=results,scope='Offline mutated copies of retained calls, no new inference/speed samples.');p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(results,indent=2))
