"""Five original metric/golden gates unchanged; strict actual-core admission only."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-primary-core-isolation';old=b/'work/worldsense-isolated-primary';slug='worldsense-primary-core-isolation';load=lambda p:json.loads(p.read_text());assert load(w/'learned-independent-audit.json')['both_original_GPU_readers_in_actual_CORE_EACH_words']==4200857600 and load(w/'admission-complete.json')['all_expected_config_rejections']
guard='''
 if call['model_id']not in ['Qwen3.5-4B','Qwen3-1.7B']:return False
 t0=call['timings']
 if 'native_text_worker'in call['backend']or'native_gateway'in t0:return False
 for cap in ['request_scope_weight_residency','packed_request_weight_residency','primary_simd_metadata','sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','sole_cooperative_prefill']:
  if cap in call['backend']:return False
 if any(key in t0 for key in ['request_weight_residency','packed_request_weight_residency','packed_request_eligible','memory_after_packed_request_cleanup']):return False
 if ('primary_core_isolated_library'in call['backend'])!=enabled:return False
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
   if after[key]-before[key]!=(encoded[oldkey]if enabled else 0):return False
   if not enabled and(before[key]!=0 or after[key]!=0):return False
   if call['model_id']=='Qwen3-1.7B'and after[key]!=before[key]:return False
 elif enabled or any(key in t0 for key in ['primary_storage_before','primary_encoded_delta']):return False
'''
files=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py'];checks={}
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
for rel in files:
 original=(old/rel).read_text();text=original.replace('worldsense-isolated-primary',slug);needle='def native_route(call,enabled,primary_control=False):\n';assert text.count(needle)==1;text=text.replace(needle,needle+guard);needle="  if (cap in call['backend'])!=enabled:return False";assert text.count(needle)==1;text=text.replace(needle,"  if cap in call['backend']:return False");ast.parse(text);p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(text)
 assert ast.dump(Normalize().visit(ast.parse(original)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse(text.replace(slug,'worldsense-isolated-primary'))),include_attributes=False),rel;checks[rel]=True
(w/'acceptance-gate-audit.json').write_text(json.dumps(dict(all_five_original_golden_numeric_metric_gate_AST_unchanged=True,files=checks,only_native_admission_changed=True,full_route_mutation_audit_pending=True,driver_sha256={rel:hashlib.sha256((w/rel).read_bytes()).hexdigest()for rel in files}),indent=2)+'\n')
for name in ['freeze.py','dependency_closure.py']:
 p=w/name;assert not p.exists();p.write_text((old/name).read_text().replace('worldsense-isolated-primary',slug))
p=w/'dependency_closure.py';text=p.read_text().replace("['off','primary','on','fault']","['off','primary','on','fault','compat56','compat56-fault']").replace('garnet-metal-worldsense-primary-core-isolation-on/direct_float_corpus_test','garnet-metal-worldsense-primary-core-isolation-compat56/direct_float_corpus_test').replace('for four builds','for six builds')
needle='    compiler=Path(subprocess.check_output';extra='''    extra_cmd=list(cmd);extra_cmd[-1]=str(w/'core_library_corpus_test.cpp');extra_text=subprocess.check_output(extra_cmd,text=True)
    (w/'core-library-corpus-preprocessor-dependencies.txt').write_text(extra_text);paths.update(Path(name)for name in dependencies(extra_text));paths.update([w/'core_library_corpus_test.cpp',r/'out/build/garnet-metal-worldsense-primary-core-isolation-on/core_library_corpus_test'])
''';assert text.count(needle)==1;text=text.replace(needle,extra+needle).replace("'preprocessor_command':cmd,","'preprocessor_command':cmd,'supplemental_preprocessor_command':extra_cmd,");ast.parse(text);p.write_text(text)
p=w/'freeze.py';text=p.read_text();start=text.index("assert {k:(flags['off'][k]");end=text.index("(w/'build-flag-comparison.json')",start)
replacement='''assert {k:(flags['off'][k],flags['on'][k])for k in flags['off']if flags['off'][k]!=flags['on'][k]}=={k:('OFF','ON')for k in ['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY']}
assert {k:(flags['primary'][k],flags['on'][k])for k in flags['on']if flags['primary'][k]!=flags['on'][k]}=={'GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY':('OFF','ON')}
for role in ['off','primary','on']:
 assert all(flags[role][key]=='OFF'for key in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY','GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS'])
''';text=text[:start]+replacement+text[end:];text=text.replace('WS54 fully-isolated wideVLM/rawtext partition; rawcandidate9flags andstoragecontrolcandidate6. Sameprimarypartition/globalretirement, differentmetadata/prefill/directupload. Faults onlyinseparatecorrectnessbuild','WS70 ordinary primary-core library isolation only, original unchanged primary fragment/preamble, RAW4flags and matchedPRIMARY1flag. No vector/full/direct/K32/cached/request flags, math/weight/geometry/runtime changes. New actual-core encoded counters timed, not completed GPU or ISA proof.')
needle="(w/'model-assets.json').write_text(json.dumps(assets,indent=2))";assert text.count(needle)==1;text=text.replace(needle,"assert json.loads((w/'model-assets.json').read_text())==assets  # Preserve frozen input bytes")
needle="shutil.copyfile(w.parent/'worldsense-request-pool/cases.json',w/'cases.json')";assert text.count(needle)==1;text=text.replace(needle,"resource_cases=json.loads((w.parent/'worldsense-request-pool/cases.json').read_text());resource_cases['repeats']=1;assert json.loads((w/'cases.json').read_text())==resource_cases  # Preserve frozen input bytes")
start=text.index("s=(w.parent/'worldsense-request-residency/controller.py')");text=text[:start]+"print('WS70 source/fourflagRAWcontrast/oneflagPRIMARYcontrast/checkpoint identities frozen',flush=True)\n";ast.parse(text);p.write_text(text)
text=(old/'controller.py').read_text().replace('worldsense-isolated-primary',slug);needle=" stop();stage(w/'before-libgarnet.dylib');start('restored-control');print('RESTORED ORIGINAL CONTROL AND LIVE ASSIGNMENTS',flush=True)";assert text.count(needle)==1;text=text.replace(needle," stop();stage(w/'before-libgarnet.dylib');subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True);print('RESTORED VALIDATED WS10 AND NORMAL NONBENCHMARK SERVER',flush=True)");assert not(w/'controller.py').exists();(w/'controller.py').write_text(text)
(w/'dispatch_prepared_resource.py').write_text((b/'work/worldsense-primary-simd-metadata/dispatch_prepared_resource.py').read_text());print('WS70 original quantitative gates and strict actual-core native routes prepared; no dispatch')
