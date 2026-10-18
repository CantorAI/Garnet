"""Original five quantitative gates unchanged; prospective SIMD-only native admission."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-primary-simd-metadata';old=b/'work/worldsense-isolated-primary';slug='worldsense-primary-simd-metadata';load=lambda p:json.loads(p.read_text())
assert load(w/'learned-independent-audit.json')['both_new_GPU_helpers_EACH_all201_original_FP32_words']==4200857600
assert load(w/'admission-complete.json')['all_expected_config_rejections']
files=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']
for rel in files:
 text=(old/rel).read_text().replace('worldsense-isolated-primary',slug)
 needle="packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']"
 guard='''
 if call['model_id']not in ['Qwen3.5-4B','Qwen3-1.7B']:return False
 if 'native_text_worker'in call['backend']or'native_gateway'in t:return False
 if ('primary_simd_metadata'in call['backend'])!=enabled:return False
 for cap in ['sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','sole_cooperative_prefill']:
  if cap in call['backend']:return False
 fields={'active_weights','allocated_bytes','uploads','decode_encoded','restore_encoded','scratch_bytes','prefill_encoded'}
 for key in ['sole_storage_before','sole_storage_after','sole_encoded_delta','dual_projection_before','dual_projection_after','dual_projection_delta']:
  group=t.get(key)
  expected=fields if key in ['sole_storage_before','sole_storage_after']else {'decode_encoded','restore_encoded','prefill_encoded'}if key=='sole_encoded_delta'else {'encoded_pairs','represented_projections'}
  if not isinstance(group,dict)or set(group)!=expected:return False
  if any(type(value)is not int or value!=0 for value in group.values()):return False
'''
 assert text.count(needle)==1;text=text.replace(needle,needle+guard)
 needle="  if (cap in call['backend'])!=enabled:return False"
 assert text.count(needle)==1;text=text.replace(needle,"  if cap in call['backend']:return False")
 if rel=='production-speed/analyze.py':
  needle=' if not enabled:\n';assert text.count(needle)==1
  text=text.replace(needle,needle+"  if call['model_id']not in ['Qwen3.5-4B','Qwen3-1.7B']or any(cap in call['backend']for cap in ['primary_full_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','primary_vector_prefill_k32_only','wide_checkpoint_storage','primary_simd_metadata','sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','sole_cooperative_prefill','native_text_worker'])or any(key in call['timings']for key in ['sole_storage_before','sole_storage_after','sole_encoded_delta','dual_projection_before','dual_projection_after','dual_projection_delta','native_gateway']):return False\n",1)
 p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();ast.parse(text);p.write_text(text)
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
checks={}
for rel in files:
 assert ast.dump(Normalize().visit(ast.parse((old/rel).read_text())),include_attributes=False)==ast.dump(Normalize().visit(ast.parse((w/rel).read_text().replace(slug,'worldsense-isolated-primary'))),include_attributes=False),rel;checks[rel]=True
(w/'acceptance-gate-audit.json').write_text(json.dumps(dict(all_five_original_golden_numeric_metric_gate_AST_unchanged=True,files=checks,only_native_admission_changed=True,full_route_mutation_audit_pending=True,driver_sha256={rel:hashlib.sha256((w/rel).read_bytes()).hexdigest()for rel in files}),indent=2)+'\n')
for name in ['freeze.py','dependency_closure.py']:
 p=w/name;assert not p.exists();p.write_text((old/name).read_text().replace('worldsense-isolated-primary',slug))
p=w/'dependency_closure.py';text=p.read_text().replace("['off','primary','on','fault']","['off','primary','on','fault','compat67']").replace("garnet-metal-worldsense-primary-simd-metadata-on/direct_float_corpus_test","garnet-metal-worldsense-primary-simd-metadata-compat67/direct_float_corpus_test").replace("for four builds","for five builds")
needle='    compiler=Path(subprocess.check_output';extra='''    extra_preprocessor_commands={}
    for source_name,role,executable in [('dual_pair_corpus_test.cpp','compat67','dual_pair_corpus_test'),('simd_metadata_corpus_test.cpp','on','simd_metadata_corpus_test')]:
        extra_cmd=list(cmd);extra_cmd[-1]=str(w/source_name);extra_text=subprocess.check_output(extra_cmd,text=True)
        (w/(source_name+'.preprocessor-dependencies.txt')).write_text(extra_text);paths.update(Path(name)for name in dependencies(extra_text));paths.update([w/source_name,r/('out/build/garnet-metal-worldsense-primary-simd-metadata-'+role)/executable]);extra_preprocessor_commands[source_name]=extra_cmd
'''
assert text.count(needle)==1;text=text.replace(needle,extra+needle).replace("'preprocessor_command':cmd,","'preprocessor_command':cmd,'supplemental_preprocessor_commands':extra_preprocessor_commands,");ast.parse(text);p.write_text(text)
p=w/'freeze.py';text=p.read_text();start=text.index("assert {k:(flags['off'][k]");end=text.index("(w/'build-flag-comparison.json')",start)
replacement='''assert {k:(flags['off'][k],flags['on'][k])for k in flags['off']if flags['off'][k]!=flags['on'][k]}=={k:('OFF','ON')for k in ['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_SIMD_METADATA']}
assert {k:(flags['primary'][k],flags['on'][k])for k in flags['on']if flags['primary'][k]!=flags['on'][k]}=={'GARNET_METAL_PRIMARY_SIMD_METADATA':('OFF','ON')}
for role in ['off','primary','on']:
 assert all(flags[role][key]=='OFF'for key in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_SOLE_INTERLEAVED_MLP','GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL','GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE','GARNET_METAL_SOLE_COOPERATIVE_PREFILL'])
'''
text=text[:start]+replacement+text[end:];text=text.replace('WS54 fully-isolated wideVLM/rawtext partition; rawcandidate9flags andstoragecontrolcandidate6. Sameprimarypartition/globalretirement, differentmetadata/prefill/directupload. Faults onlyinseparatecorrectnessbuild','WS68 original PRIMARY/WIDE/LAZY plus device SIMD metadata sharing. RAW4flag contrast; matchedPRIMARY1flag contrast. Original payload and floating arithmetic unchanged, no later rejected profiles enabled. Compatibility67 correctness only.')
needle="(w/'model-assets.json').write_text(json.dumps(assets,indent=2))";assert text.count(needle)==1;text=text.replace(needle,"assert json.loads((w/'model-assets.json').read_text())==assets  # Preserve previously frozen input bytes")
needle="shutil.copyfile(w.parent/'worldsense-request-pool/cases.json',w/'cases.json')";assert text.count(needle)==1;text=text.replace(needle,"resource_cases=json.loads((w.parent/'worldsense-request-pool/cases.json').read_text());resource_cases['repeats']=1;assert json.loads((w/'cases.json').read_text())==resource_cases  # Exact original resource/functional repeat count; preserve frozen input")
start=text.index("s=(w.parent/'worldsense-request-residency/controller.py')");text=text[:start]+"print('WS68 source/fourflagRAWcontrast/oneflagPRIMARYcontrast/checkpoint identities frozen',flush=True)\n";ast.parse(text);p.write_text(text)
text=(old/'controller.py').read_text().replace('worldsense-isolated-primary',slug);needle=" stop();stage(w/'before-libgarnet.dylib');start('restored-control');print('RESTORED ORIGINAL CONTROL AND LIVE ASSIGNMENTS',flush=True)";assert text.count(needle)==1;text=text.replace(needle," stop();stage(w/'before-libgarnet.dylib');subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True);print('RESTORED VALIDATED WS10 AND NORMAL NONBENCHMARK SERVER',flush=True)");(w/'controller.py').write_text(text)
print('Prepared original quantitative gates with SIMD-only native route and exact profile contrasts; no dispatch.')
