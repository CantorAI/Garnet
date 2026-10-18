from pathlib import Path
import json,ast,copy,hashlib,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-validated-text-worker';rels=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py'];all_results={}
for rel in rels:
 text=(w/rel).read_text();tree=ast.parse(text);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(w/rel),'exec'),ns);route=ns['native_route'];positives=0;negatives=0
 for mode in ['off','primary','on']:
  if rel=='production-speed/analyze.py'and mode!='on':continue
  records=json.loads((w/('api-state-progress-'+mode+'.json')).read_text())['records'];assert len(records)==5
  for row in records:
   c=row['call'];enabled=mode=='on';primary=mode=='primary';assert route(c,enabled,primary),(rel,mode,c['model_id']);positives+=1
   def reject(value):
    global negatives
    assert not route(value,enabled,primary),(rel,mode,'mutation accepted');negatives+=1
   caps=['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults','primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library','native_text_worker']
   for cap in caps:
    bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace(cap,'')if cap in bad['backend']else bad['backend']+' '+cap;reject(bad)
   bad=copy.deepcopy(c);bad['timings']['primary_storage_after']['metadata_bytes']+=4;reject(bad)
   if mode=='off':continue
   for key in ['cohort_process_count','metal_allocated_bytes','task_physical_footprint_bytes','metal_recommended_working_set_bytes']:
    bad=copy.deepcopy(c);bad['timings']['memory_after_decode'][key]+=1;reject(bad)
   bad=copy.deepcopy(c);bad['timings']['memory_after_decode']['process_samples'].append(copy.deepcopy(bad['timings']['memory_after_decode']['process_samples'][0]));reject(bad)
   for key in ['decode_encoded','prefill_encoded','head_encoded','vector_decode_encoded','vector_head_encoded']:
    bad=copy.deepcopy(c);bad['timings']['native_gateway']['parent_primary_encoded_delta'][key]+=1;reject(bad)
   bad=copy.deepcopy(c);bad['ttft_seconds']+=.001;reject(bad)
   bad=copy.deepcopy(c);bad['timings']['native_gateway']['parent_primary_storage_after']['active_weights']=398;reject(bad)
   for cap in caps:
    bad=copy.deepcopy(c);g=bad['timings']['native_gateway'];g['parent_backend']=g['parent_backend'].replace(cap,'')if cap in g['parent_backend']else g['parent_backend']+' '+cap;reject(bad)
   if mode!='off':
    mutations=[(['native_gateway','parent_backend'],'wrong-parent'),(['native_gateway','child','model_backend','library_path'],'/wrong/libgarnet.dylib'),(['native_gateway','child','model_backend','library_sha256'],'wrong'),(['native_gateway','child','model_backend','loaded_garnet_images'],2),(['native_gateway','child','model_backend','package_abi'],0),(['native_gateway','child','model_backend','expected_original_source_root'],'/wrong/source'),(['native_gateway','child','decoder_source'],'/wrong/decoder.py'),(['native_gateway','child','serving_source'],'/wrong/metal_serving.py'),(['native_gateway','child','raw_counter_policy'],'wrong'),(['native_gateway','child','raw_counter_origin'],'wrong')]
    if c['model_id']=='Qwen3-1.7B':mutations += [(['counter_origin'],'wrong'),(['serving_source'],'/wrong/metal_serving.py'),(['lazy_default_retired_bytes'],False)]
    for path,value in mutations:
     bad=copy.deepcopy(c);target=bad['timings']
     for key in path[:-1]:target=target[key]
     target[path[-1]]=value;reject(bad)
   if c['model_id']=='Qwen3-1.7B':
    for key,value in [('birth_id','wrong'),('infer_calls',123456),('models',[])]:
     bad=copy.deepcopy(c);bad['timings']['native_gateway']['child'][key]=value;reject(bad)
    for key,value in [('parent_pid',-1),('watchdog_active',False),('library_path','/wrong/libgarnet.dylib'),('source_root','/wrong/source')]:
     bad=copy.deepcopy(c);bad['timings']['native_gateway']['child']['support'][key]=value;reject(bad)
 if rel=='production-speed/analyze.py':
  legacy=json.loads((w.parent/'worldsense-packed-only-defaults/production-speed/control-1.json').read_text());n=0
  for item in legacy['items']:
   for c in item['result']['model_response'].values():
    if isinstance(c,dict)and c.get('status')=='ok':assert route(c,False);n+=1
  assert n==10
 all_results[rel]=dict(actual_current_full_API_calls=positives,malformed_native_admission_mutations_rejected=negatives,legacy_WS10_10calls_pass=rel=='production-speed/analyze.py')
p=w/'acceptance-gate-audit.json';d=json.loads(p.read_text());d.update(full_route_mutation_audit_pending=False,full_route_mutation_audit_pass=True,native_route_audit=all_results,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(all_results,indent=2))
