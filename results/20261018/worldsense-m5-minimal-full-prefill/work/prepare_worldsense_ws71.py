"""Five original metric/golden gates unchanged; strict actual-core admission only."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-minimal-full-prefill';old=b/'work/worldsense-isolated-primary';slug='worldsense-minimal-full-prefill';load=lambda p:json.loads(p.read_text());assert load(w/'learned-independent-audit.json')['all_five_readers_in_actual_FULL_candidate'] is True and load(w/'admission-complete.json')['all_expected_config_rejections']
guard='''
 if call['model_id']not in ['Qwen3.5-4B','Qwen3-1.7B']:return False
 t0=call['timings']
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
'''
files=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py'];checks={}
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
for rel in files:
 original=(old/rel).read_text();text=original.replace('worldsense-isolated-primary',slug);needle='def native_route(call,enabled,primary_control=False):\n';assert text.count(needle)==1;entry_needle=needle;needle="  if (cap in call['backend'])!=enabled:return False";assert text.count(needle)==1;text=text.replace(needle,"  if cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','primary_full_isolated_library']:\n   if (cap in call['backend'])!=enabled:return False\n  elif cap in call['backend']:return False");text=text.replace(entry_needle,entry_needle+guard);ast.parse(text);p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(text)
 assert ast.dump(Normalize().visit(ast.parse(original)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse(text.replace(slug,'worldsense-isolated-primary'))),include_attributes=False),rel;checks[rel]=True
(w/'acceptance-gate-audit.json').write_text(json.dumps(dict(all_five_original_golden_numeric_metric_gate_AST_unchanged=True,files=checks,only_native_admission_changed=True,full_route_mutation_audit_pending=True,driver_sha256={rel:hashlib.sha256((w/rel).read_bytes()).hexdigest()for rel in files}),indent=2)+'\n')
for name in ['freeze.py','dependency_closure.py']:
 p=w/name;assert not p.exists();p.write_text((old/name).read_text().replace('worldsense-isolated-primary',slug))
p=w/'dependency_closure.py';text=p.read_text().replace("['off','primary','on','fault']","['off','primary','on','fault','compat56','compat56-fault']").replace('garnet-metal-worldsense-minimal-full-prefill-on/direct_float_corpus_test','garnet-metal-worldsense-minimal-full-prefill-compat56/direct_float_corpus_test').replace('for four builds','for six builds')
text=text.replace('for six builds','for six builds; actual candidate full five-reader corpus')
ast.parse(text);p.write_text(text)
p=w/'freeze.py';text=p.read_text();start=text.index("assert {k:(flags['off'][k]");end=text.index("(w/'build-flag-comparison.json')",start)
replacement='''assert {k:(flags['off'][k],flags['on'][k])for k in flags['off']if flags['off'][k]!=flags['on'][k]}=={k:('OFF','ON')for k in ['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']}
assert {k:(flags['primary'][k],flags['on'][k])for k in flags['on']if flags['primary'][k]!=flags['on'][k]}=={key:('OFF','ON')for key in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']}
for role in ['off','primary','on']:
 assert all(flags[role][key]=='OFF'for key in ['GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY','GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY','GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS'])
''';text=text[:start]+replacement+text[end:];text=text.replace('WS54 fully-isolated wideVLM/rawtext partition; rawcandidate9flags andstoragecontrolcandidate6. Sameprimarypartition/globalretirement, differentmetadata/prefill/directupload. Faults onlyinseparatecorrectnessbuild','WS71 existing minimalFULL+vector profile, ZERO1721sourcechanges, RAW6flags and matchedPRIMARY3flags. Direct/K32/cached/rawrequest/CORE OFF. Actual three core counters zero; same native telemetry included, no isolated causal or GPU ISA proof.')
needle="(w/'model-assets.json').write_text(json.dumps(assets,indent=2))";assert text.count(needle)==1;text=text.replace(needle,"assert json.loads((w/'model-assets.json').read_text())==assets  # Preserve frozen input bytes")
needle="shutil.copyfile(w.parent/'worldsense-request-pool/cases.json',w/'cases.json')";assert text.count(needle)==1;text=text.replace(needle,"resource_cases=json.loads((w.parent/'worldsense-request-pool/cases.json').read_text());resource_cases['repeats']=1;assert json.loads((w/'cases.json').read_text())==resource_cases  # Preserve frozen input bytes")
start=text.index("s=(w.parent/'worldsense-request-residency/controller.py')");text=text[:start]+"print('WS71 source/sixflagRAWcontrast/threeflagPRIMARYcontrast/checkpoint identities frozen',flush=True)\n";ast.parse(text);p.write_text(text)
text=(old/'controller.py').read_text().replace('worldsense-isolated-primary',slug);needle=" stop();stage(w/'before-libgarnet.dylib');start('restored-control');print('RESTORED ORIGINAL CONTROL AND LIVE ASSIGNMENTS',flush=True)";assert text.count(needle)==1;text=text.replace(needle," stop();stage(w/'before-libgarnet.dylib');subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True);print('RESTORED VALIDATED WS10 AND NORMAL NONBENCHMARK SERVER',flush=True)");assert not(w/'controller.py').exists();(w/'controller.py').write_text(text)
(w/'dispatch_prepared_resource.py').write_text((b/'work/worldsense-primary-simd-metadata/dispatch_prepared_resource.py').read_text());print('WS71 original quantitative gates and strict actual-core native routes prepared; no dispatch')
