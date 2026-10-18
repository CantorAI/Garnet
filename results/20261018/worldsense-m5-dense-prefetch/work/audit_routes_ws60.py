"""Offline real-record positive and malformed-copy native admission audit."""
from pathlib import Path
import json,ast,copy,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-dense-prefetch'
rels=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py'];results={}
for rel in rels:
 tree=ast.parse((w/rel).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');ns={}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'current native route','exec'),ns);route=ns['native_route'];positive=0;negative=0
 for mode in ['off','primary','on']:
  if rel=='production-speed/analyze.py'and mode!='on':continue
  records=json.loads((w/('api-state-progress-'+mode+'.json')).read_text())['records'];assert len(records)==5
  for row in records:
   c=row['call'];enabled=mode=='on';primary=mode=='primary';assert route(c,enabled,primary),(rel,mode,c['model_id']);positive+=1
   def reject(value):
    global negative
    assert not route(value,enabled,primary),(rel,mode,'invalid copied fixture accepted');negative+=1
   for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults','primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library','dense_raw_decode_prefetch']:
    bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace(cap,'')if cap in bad['backend']else bad['backend']+' '+cap;reject(bad)
   for value in [None,-1,True,False,'1',float('nan')]:
    bad=copy.deepcopy(c);bad['timings']['dense_prefetch_encoded_delta']=value;reject(bad)
   bad=copy.deepcopy(c);bad['timings'].pop('dense_prefetch_encoded_delta');reject(bad)
   bad=copy.deepcopy(c);bad['timings']['dense_prefetch_encoded_delta']=0 if c['timings']['dense_prefetch_encoded_delta']>0 else 1;reject(bad)
   bad=copy.deepcopy(c);bad['backend']+=' native_text_worker';reject(bad)
   bad=copy.deepcopy(c);bad['timings']['native_gateway']={};reject(bad)
   bad=copy.deepcopy(c);bad['model_id']='unknown';reject(bad)
   bad=copy.deepcopy(c);bad['timings']['primary_storage_after']['metadata_bytes']+=4;reject(bad)
   bad=copy.deepcopy(c);bad['timings']['memory_after_decode']['task_info_status']=1;reject(bad)
   bad=copy.deepcopy(c);bad['timings']['lazy_default_rehydrated_bytes']+=1;reject(bad)
   if c['model_id']=='Qwen3-1.7B'and mode!='off':
    bad=copy.deepcopy(c);bad['timings']['primary_encoded_delta']['decode_encoded']+=1;reject(bad)
 legacy_count=0
 if rel=='production-speed/analyze.py':
  data=json.loads((w.parent/'worldsense-isolated-primary/production-speed/control-1.json').read_text())
  for item in data['items']:
   for c in item['result']['model_response'].values():
    if not isinstance(c,dict)or c.get('status')!='ok':continue
    assert route(c,False);legacy_count+=1
    for cap in ['dense_raw_decode_prefetch','native_text_worker']:
     bad=copy.deepcopy(c);bad['backend']+=' '+cap;assert not route(bad,False);negative+=1
    for field in ['native_gateway','dense_prefetch_encoded_delta']:
     bad=copy.deepcopy(c);bad['timings'][field]=0;assert not route(bad,False);negative+=1
  assert legacy_count==10
 results[rel]=dict(actual_full_API_calls=positive,malformed_copied_fixtures_rejected=negative,actual_legacy_WS10_calls=legacy_count)
p=w/'acceptance-gate-audit.json';d=json.loads(p.read_text());d.update(full_route_mutation_audit_pending=False,full_route_mutation_audit_pass=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_route_audit=results,scope='Offline copies of already retained actual calls; not additional model inference or independent speed samples.')
p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(results,indent=2))
