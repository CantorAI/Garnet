from pathlib import Path
import ast,json
w=Path('work/worldsense-wide-cached-prefill')
route='''def native_route(call,enabled,primary_control=False):
 packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:
  if (cap in call['backend'])!=packed:return False
 for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata']:
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
'''
oldbranch=""" if not enabled:
  return 'dense_prefix_cache'in call['backend']and not any(k in call['backend']for k in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','primary_vector_prefill','primary_prefill_isolated_library','lazy_resident_defaults','cached_matrix_metadata','wide_primary_storage','direct_float_weight_upload'])and not any(k in call['timings']for k in ['primary_storage_after','primary_encoded_delta','memory_after_decode'])
"""
for rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
 p=w/rel;s=p.read_text();tree=ast.parse(s);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');lines=s.splitlines(keepends=True)
 replacement=route
 if rel=='production-speed/analyze.py':replacement=route.replace(" packed=enabled or primary_control",oldbranch+" packed=enabled or primary_control",1)
 s=''.join(lines[:fn.lineno-1])+replacement+''.join(lines[fn.end_lineno:]);ast.parse(s);p.write_text(s)
# Freeze checks correspond exactly to predeclared eight/raw and five/storage contrasts.
p=w/'freeze.py';s=p.read_text();s=s.replace("'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),", "'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),'GARNET_METAL_WIDE_PRIMARY_STORAGE':('OFF','ON'),",1)
a=s.index("assert all(flags[m][cap]");b=s.index("\n(w/'build-flag",a)
s=s[:a]+'''assert all(flags[m][cap]==('OFF'if m=='off'else'ON')for m in ['off','primary','on']for cap in ['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS'])
assert all(flags[m]['GARNET_METAL_CACHED_MATRIX_METADATA']==('ON'if m=='on'else'OFF')for m in ['off','primary','on'])
assert {k:(flags['primary'][k],flags['on'][k])for k in flags['on']if flags['primary'][k]!=flags['on'][k]}=={k:('OFF','ON')for k in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_CACHED_MATRIX_METADATA']}
'''+s[b:]
s=s.replace('Zero-source-deltaWS51; rawcandidate7flags andprimary-STORAGE-controlcandidate6. Sameprimaryrepresentation, differentlifecycle/metadata/prefill/directupload.','WS53 wideVLM/rawtext partition; rawcandidate8flags andstoragecontrolcandidate5. Sameprimarypartition/globalretirement, differentmetadata/prefill/directupload.')
p.write_text(s)
# Actual deployed control is the first speed stage, generated from root independently.
p=w/'production-speed/prepare.py';s=p.read_text().replace("p/'speed/analysis.json'","p/'resource-analysis.json'")
s=s.replace("for name in ['starting-source-manifest.json','cases.json']:shutil.copy2(p/'speed'/name,w/name)","shutil.copy2(p/'starting-source-manifest.json',w/'starting-source-manifest.json')\nc=json.loads((p/'cases.json').read_text());c['repeats']=4;(w/'cases.json').write_text(json.dumps(c,indent=2))")
s=s.replace("t=(p/'speed/controller.py').read_text().replace('worldsense-wide-cached-prefill/speed','worldsense-wide-cached-prefill/production-speed')",'''t=(p/'controller.py').read_text().replace("work/worldsense-wide-cached-prefill'","work/worldsense-wide-cached-prefill/production-speed'").replace("stage(w/'before-libgarnet.dylib')","stage(w.parent/'before-libgarnet.dylib')").replace("str(w/'wait_cantor.py')","str(w.parent/'wait_cantor.py')")''')
s=s.replace("assert all(sha(control/rel)==h for rel,h in manifest.items())","assert len(manifest)==1706 and all(sha(control/rel)==h for rel,h in manifest.items())")
p.write_text(s)
p=w/'primary-speed/prepare.py';s=p.read_text().replace("assert json.loads((p/'resource-analysis.json').read_text())['passed']", "assert json.loads((p/'resource-analysis.json').read_text())['passed']\nassert json.loads((p/'production-speed/analysis.json').read_text())['passed']")
s=s.replace("'GARNET_METAL_LAZY_RESIDENT_DEFAULTS':('OFF','ON'),",'').replace('Sixflags(vector/isolation/directupload/K32/lazy/cached): sameprimaryweightstorage, differentlifecycle/metadata/prefill; not eager/eager','Fiveflags(vector/isolation/directupload/K32/cached): samewideVLM/rawtextpartition andglobalretirement, differentmetadata/prefill/directupload').replace('prepared_after_resource_pass','prepared_after_actual_WS10_speed_pass')
p.write_text(s)
p=w/'speed/prepare.py';s=p.read_text().replace("assert json.loads((p/'resource-analysis.json').read_text())['passed']", "assert json.loads((p/'resource-analysis.json').read_text())['passed']\nassert json.loads((p/'production-speed/analysis.json').read_text())['passed']");p.write_text(s)
# Route adversarial audit checks lifecycle against global primary policy, cache against candidate.
p=w/'audit_gates.py';s=p.read_text().replace("'primary_u32_metadata']:","'primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:")
s=s.replace(",'lazy_resident_defaults','cached_matrix_metadata']:",",'cached_matrix_metadata']:")
s=s.replace("0 if enabled and field!='lazy_default_rehydrated_bytes'else 1", "0 if packed and field!='lazy_default_rehydrated_bytes'else 1")
p.write_text(s)
p=w/'prepare_cohort.py';s=p.read_text().replace("['zero_source_delta_WS51']","['wide_admission_scope_verified']");p.write_text(s)
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
(w/'benchmark-driver-adaptation.json').write_text(json.dumps({'stage_order':json.loads((w/'protocol.json').read_text())['stage_order'],'numeric_thresholds_unchanged':True,'raw_flags':8,'storage_control_flags':5,'text_primary_deltas_must_be_zero':True,'old_WS10_missing_memory_is_null':True,'scope':'Pre-timing driver adaptation; actual API adversarial/AST audit must pass before dispatch.'},indent=2)+'\n')
print('WS53 benchmark policies/stage order adapted; actual-call audits still required')
