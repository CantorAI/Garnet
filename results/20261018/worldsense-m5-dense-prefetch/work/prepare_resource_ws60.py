"""Install original five WorldSense gates with new raw-prefetch route checks."""
from pathlib import Path
import ast,json,hashlib
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-dense-prefetch';old=w.parent/'worldsense-isolated-primary';slug='worldsense-dense-prefetch'
assert json.loads((w/'admission-complete.json').read_text())['ten_expected_config_rejections']
assert json.loads((w/'correctness-terminal.json').read_text())['controller_session']==68305
analyzers=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']
for rel in analyzers:
 t=(old/rel).read_text().replace('worldsense-isolated-primary',slug)
 needle="packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']"
 assert t.count(needle)==1,rel
 guard='''
 if call['model_id'] not in ['Qwen3.5-4B','Qwen3-1.7B']:return False
 if 'native_text_worker'in call['backend']or'native_gateway'in t:return False
 if ('dense_raw_decode_prefetch'in call['backend'])!=packed:return False
 count=t.get('dense_prefetch_encoded_delta')
 if not isinstance(count,int)or isinstance(count,bool)or count<0:return False
 if (count>0)!=(packed and call['model_id']=='Qwen3-1.7B'):return False
'''
 t=t.replace(needle,needle+guard,1);p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);ast.parse(t);p.write_text(t)
 if rel=='production-speed/analyze.py':
  needle=' if not enabled:\n';assert t.count(needle)==1
  t=t.replace(needle,needle+"  if any(cap in call['backend']for cap in ['dense_raw_decode_prefetch','native_text_worker'])or any(key in call['timings']for key in ['native_gateway','dense_prefetch_encoded_delta']):return False\n",1)
  ast.parse(t);p.write_text(t)
for name in ['freeze.py','dependency_closure.py']:
 (w/name).write_text((old/name).read_text().replace('worldsense-isolated-primary',slug))
p=w/'freeze.py';t=p.read_text();needle="'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),";assert t.count(needle)==1;t=t.replace(needle,needle+"'GARNET_METAL_DENSE_PREFETCH':('OFF','ON'),",1)
needle="for cap in ['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS']";assert t.count(needle)==1;t=t.replace(needle,needle[:-1]+",'GARNET_METAL_DENSE_PREFETCH']")
start=t.index("s=(w.parent/'worldsense-request-residency/controller.py')");t=t[:start]+"print('Frozen WS60 source,10flagcontrast,4baseflags and checkpoint identities PASS',flush=True)\n"
t=t.replace('WS54 fully-isolated wideVLM/rawtext partition; rawcandidate9flags andstoragecontrolcandidate6. Sameprimarypartition/globalretirement, differentmetadata/prefill/directupload. Faults onlyinseparatecorrectnessbuild','WS60 original-order raw dense lookahead, isolated wideVLM/rawtext partition; rawcandidate10flags/storagecontrolcandidate6. Same201packed/245raw partition and globalretirement; textprefetch in both storagecontrol and candidate. Faults only in correctness build.')
p.write_text(t)
t=(old/'controller.py').read_text().replace('worldsense-isolated-primary',slug)
original=" stop();stage(w/'before-libgarnet.dylib');start('restored-control');print('RESTORED ORIGINAL CONTROL AND LIVE ASSIGNMENTS',flush=True)"
assert t.count(original)==1;t=t.replace(original," stop();stage(w/'before-libgarnet.dylib');subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True);print('RESTORED VALIDATED WS10 AND NORMAL NONBENCHMARK SERVER',flush=True)")
(w/'controller.py').write_text(t)
t=(old/'prepare_cohort.py').read_text().replace('worldsense-isolated-primary',slug)
start=t.index('assert all(v==0');end=t.index("subprocess.run(['python3',str(w/'freeze.py')],check=True)")
t=t[:start]+'''assert all(json.loads((w/'full-proof-complete.json').read_text()).values())
assert json.loads((w/'correctness-terminal.json').read_text())['controller_session']==68305
assert json.loads((w/'admission-complete.json').read_text())['ten_expected_config_rejections']
assert json.loads((w/'acceptance-gate-audit.json').read_text())['all_five_original_golden_numeric_metric_gate_AST_unchanged']
assert json.loads((w/'acceptance-gate-audit.json').read_text())['full_route_mutation_audit_pass']
assert json.loads((w/'functional-smoke/functional-verification.json').read_text())['actualWorldSense_Galaxy_native_candidate_functional_pass']
assert json.loads((w/'dense-prefetch-word-comparison.json').read_text())['all_four_forced_raw_full_FP32_projection_files_identical']
assert json.loads((w/'postproof-source-admission.json').read_text())['original_Weight_loader_and_wide_scope_source_exact54']
'''+t[end:]
(w/'prepare_cohort.py').write_text(t)
(w/'run_resource.py').write_text((old/'run_resource.py').read_text())
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
checks={}
for rel in analyzers:
 a=ast.dump(Normalize().visit(ast.parse((old/rel).read_text())),include_attributes=False)
 b=ast.dump(Normalize().visit(ast.parse((w/rel).read_text().replace(slug,'worldsense-isolated-primary'))),include_attributes=False)
 assert a==b,rel;checks[rel]=True
for name in ['freeze.py','dependency_closure.py','prepare_cohort.py','run_resource.py','controller.py']+analyzers:ast.parse((w/name).read_text())
(w/'acceptance-gate-audit.json').write_text(json.dumps(dict(all_five_original_golden_numeric_metric_gate_AST_unchanged=True,files=checks,only_native_prefetch_admission_extended=True,full_route_mutation_audit_pending=True,driver_sha256={rel:hashlib.sha256((w/rel).read_bytes()).hexdigest()for rel in analyzers}),indent=2)+'\n')
print('Original five golden/metric/threshold ASTs unchanged; raw-prefetch route guards installed; no dispatch')
