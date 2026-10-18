"""Prepare unchanged five quantitative gates with strict packed-request admission only."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-packed-request-residency';old=b/'work/worldsense-isolated-primary';slug='worldsense-packed-request-residency';load=lambda p:json.loads(p.read_text())
assert load(w/'learned-independent-audit.json')['both_original_GPU_readers_under_actual_lease_EACH_words']==4200857600
assert load(w/'admission-complete.json')['all_expected_config_rejections']
guard='''
 if call['model_id']not in ['Qwen3.5-4B','Qwen3-1.7B']:return False
 if 'native_text_worker'in call['backend']or'native_gateway'in call['timings']:return False
 for cap in ['request_scope_weight_residency','primary_simd_metadata','sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','sole_cooperative_prefill']:
  if cap in call['backend']:return False
 if 'request_weight_residency'in call['timings']:return False
 if ('packed_request_weight_residency'in call['backend'])!=enabled:return False
 t0=call['timings'];newkeys=['packed_request_eligible','packed_request_weight_residency','memory_after_packed_request_cleanup']
 if not enabled:
  if any(key in t0 for key in newkeys):return False
 else:
  if any(key not in t0 for key in newkeys):return False
  eligible=call['model_id']=='Qwen3.5-4B'
  if type(t0['packed_request_eligible'])is not bool or t0['packed_request_eligible']!=eligible:return False
  q=t0['packed_request_weight_residency'];memory=t0['memory_after_packed_request_cleanup']
  fields={'calls','admitted','fallback','ends','last_bytes','last_count','active_bytes','global_used_bytes','limit_bytes','attempted_bytes','attempted_count','reason','set_allocation_count','owner_count','coverage_cache_hits','coverage_checks','coverage_fallbacks','last_coverage_graphs'}
  flags={'enabled','scope_open','attached','requested'}
  if not isinstance(q,dict)or set(q)!=fields|flags:return False
  if any(type(q[k])is not int or q[k]<0 for k in fields)or any(type(q[k])is not bool for k in flags):return False
  if not q['enabled']or q['scope_open']or q['attached']or q['requested']or q['limit_bytes']!=8589934592:return False
  if any(q[k]!=0 for k in ['active_bytes','global_used_bytes','set_allocation_count','owner_count']):return False
  if not isinstance(memory,dict)or type(memory.get('task_info_status'))is not int or memory['task_info_status']!=0:return False
  for key in ['metal_allocated_bytes','task_physical_footprint_bytes','task_resident_bytes','task_compressed_bytes','task_internal_bytes','task_virtual_bytes']:
   if type(memory.get(key))is not int or memory[key]<0:return False
  if eligible:
   if not(q['calls']==q['admitted']==q['ends']>0)or q['fallback']!=0 or q['reason']!=1:return False
   if q['last_count']!=402 or q['attempted_count']!=402:return False
   if q['last_bytes']!=q['attempted_bytes']or q['last_bytes']!=t0['primary_storage_after']['allocated_bytes']or not(6402163712<=q['last_bytes']<=q['limit_bytes']):return False
   if q['coverage_cache_hits']<=0 or q['coverage_checks']<=0 or q['coverage_fallbacks']!=0 or not(33<q['last_coverage_graphs']<=256):return False
  elif any(q[k]!=0 for k in fields-{'limit_bytes'}):return False
'''
files=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py'];checks={}
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
for rel in files:
 original=(old/rel).read_text();text=original.replace('worldsense-isolated-primary',slug)
 needle='def native_route(call,enabled,primary_control=False):\n';assert text.count(needle)==1;text=text.replace(needle,needle+guard)
 needle="  if (cap in call['backend'])!=enabled:return False";assert text.count(needle)==1;text=text.replace(needle,"  if cap in call['backend']:return False")
 ast.parse(text);p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),rel;p.write_text(text)
 assert ast.dump(Normalize().visit(ast.parse(original)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse(text.replace(slug,'worldsense-isolated-primary'))),include_attributes=False),rel
 checks[rel]=True
(w/'acceptance-gate-audit.json').write_text(json.dumps(dict(all_five_original_golden_numeric_metric_gate_AST_unchanged=True,files=checks,only_native_admission_changed=True,full_route_mutation_audit_pending=True,driver_sha256={rel:hashlib.sha256((w/rel).read_bytes()).hexdigest()for rel in files}),indent=2)+'\n')
for name in ['freeze.py','dependency_closure.py']:
 p=w/name;assert not p.exists();p.write_text((old/name).read_text().replace('worldsense-isolated-primary',slug))
p=w/'dependency_closure.py';text=p.read_text().replace("['off','primary','on','fault']","['off','primary','on','fault','compat56','compat56-fault']").replace('garnet-metal-worldsense-packed-request-residency-on/direct_float_corpus_test','garnet-metal-worldsense-packed-request-residency-compat56/direct_float_corpus_test').replace('for four builds','for six builds')
needle='    compiler=Path(subprocess.check_output';extra='''    extra_cmd=list(cmd);extra_cmd[-1]=str(w/'packed_lease_corpus_test.cpp');extra_text=subprocess.check_output(extra_cmd,text=True)
    (w/'packed-lease-corpus-preprocessor-dependencies.txt').write_text(extra_text);paths.update(Path(name)for name in dependencies(extra_text));paths.update([w/'packed_lease_corpus_test.cpp',r/'out/build/garnet-metal-worldsense-packed-request-residency-on/packed_lease_corpus_test'])
'''
assert text.count(needle)==1;text=text.replace(needle,extra+needle).replace("'preprocessor_command':cmd,","'preprocessor_command':cmd,'supplemental_preprocessor_command':extra_cmd,");ast.parse(text);p.write_text(text)
p=w/'freeze.py';text=p.read_text();start=text.index("assert {k:(flags['off'][k]");end=text.index("(w/'build-flag-comparison.json')",start)
replacement='''assert {k:(flags['off'][k],flags['on'][k])for k in flags['off']if flags['off'][k]!=flags['on'][k]}=={k:('OFF','ON')for k in ['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY']}
assert {k:(flags['primary'][k],flags['on'][k])for k in flags['on']if flags['primary'][k]!=flags['on'][k]}=={'GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY':('OFF','ON')}
for role in ['off','primary','on']:
 assert all(flags[role][key]=='OFF'for key in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY','GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS','GARNET_METAL_PACKED_REQUEST_RESIDENCY_TEST_FAULTS'])
'''
text=text[:start]+replacement+text[end:];text=text.replace('WS54 fully-isolated wideVLM/rawtext partition; rawcandidate9flags andstoragecontrolcandidate6. Sameprimarypartition/globalretirement, differentmetadata/prefill/directupload. Faults onlyinseparatecorrectnessbuild','WS69 original PRIMARY/WIDE/LAZY plus bounded packed payload/metadata request hints. RAW4flag and matchedPRIMARY1flag contrast. Shaders/math/weights unchanged; automatic raw handling and no extra companion native calls. Allocations/hints do not prove physical residency or speed.')
needle="(w/'model-assets.json').write_text(json.dumps(assets,indent=2))";assert text.count(needle)==1;text=text.replace(needle,"assert json.loads((w/'model-assets.json').read_text())==assets  # Preserve frozen input bytes")
needle="shutil.copyfile(w.parent/'worldsense-request-pool/cases.json',w/'cases.json')";assert text.count(needle)==1;text=text.replace(needle,"resource_cases=json.loads((w.parent/'worldsense-request-pool/cases.json').read_text());resource_cases['repeats']=1;assert json.loads((w/'cases.json').read_text())==resource_cases  # Preserve frozen input bytes")
start=text.index("s=(w.parent/'worldsense-request-residency/controller.py')");text=text[:start]+"print('WS69 source/fourflagRAWcontrast/oneflagPRIMARYcontrast/checkpoint identities frozen',flush=True)\n";ast.parse(text);p.write_text(text)
text=(old/'controller.py').read_text().replace('worldsense-isolated-primary',slug);needle=" stop();stage(w/'before-libgarnet.dylib');start('restored-control');print('RESTORED ORIGINAL CONTROL AND LIVE ASSIGNMENTS',flush=True)";assert text.count(needle)==1;text=text.replace(needle," stop();stage(w/'before-libgarnet.dylib');subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True);print('RESTORED VALIDATED WS10 AND NORMAL NONBENCHMARK SERVER',flush=True)");assert not(w/'controller.py').exists();(w/'controller.py').write_text(text)
text=(b/'work/worldsense-primary-simd-metadata/dispatch_prepared_resource.py').read_text();assert not(w/'dispatch_prepared_resource.py').exists();(w/'dispatch_prepared_resource.py').write_text(text)
print('WS69 original quantitative gates prepared with strict packed-allocation/cleanup/companion admission; no dispatch.')
