"""Install reviewed same-metric WorldSense drivers, never dispatch timing."""
from pathlib import Path
import ast,json,hashlib
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-thread-qos';old=w.parent/'worldsense-isolated-primary';slug='worldsense-thread-qos'
assert json.loads((w/'admission-complete.json').read_text())['ten_expected_config_rejections']
route=(root/'work/worldsense-opt/ws59_native_route.py').read_text();route=route[route.index('def native_route('):].rstrip()+'\n'
analyzers=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']
for rel in analyzers:
 text=(old/rel).read_text().replace('worldsense-isolated-primary',slug);tree=ast.parse(text);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');lines=text.splitlines(keepends=True);replacement=route
 if rel=='production-speed/analyze.py':
  # Preserve original legacy WS10 admission exactly; its telemetry is unavailable.
  original=ast.get_source_segment(text,fn);prefix=original[original.index(' if not enabled:'):original.index(' packed=')]
  replacement=route.replace(' import math\n',prefix+' import math\n',1)
 text=''.join(lines[:fn.lineno-1])+replacement+''.join(lines[fn.end_lineno:]);p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
for name in ['freeze.py','dependency_closure.py']:(w/name).write_text((old/name).read_text().replace('worldsense-isolated-primary',slug).replace('correctness-source-manifest.json','gateway-source-manifest.json'))
p=w/'freeze.py';t=p.read_text();needle="'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),";assert t.count(needle)==1;t=t.replace(needle,needle+"'GARNET_METAL_NATIVE_TEXT_WORKER':('OFF','ON'),'GARNET_METAL_NATIVE_TEXT_WORKER_QOS':('OFF','ON'),",1);needle="for cap in ['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS']";assert t.count(needle)==1;t=t.replace(needle,needle[:-1]+",'GARNET_METAL_NATIVE_TEXT_WORKER','GARNET_METAL_NATIVE_TEXT_WORKER_QOS']")
start=t.index("s=(w.parent/'worldsense-request-residency/controller.py')");t=t[:start]+"print('Frozen WS59 source,11flagcontrast,5baseflags/nativeRAWchild and checkpoint identities PASS',flush=True)\n"
t=t.replace('WS54 fully-isolated wideVLM/rawtext partition; rawcandidate9flags andstoragecontrolcandidate6. Sameprimarypartition/globalretirement, differentmetadata/prefill/directupload. Faults onlyinseparatecorrectnessbuild','WS59 packedVLM/nativeRAWtext process isolation; rawcandidate11flags/storagecontrolcandidate6. Same-source RAW child included in all candidate process-set memory. Faults only in correctness build.')
p.write_text(t)
p=w/'controller.py';t=(w.parent/'worldsense-packed-only-defaults/controller.py').read_text().replace('worldsense-packed-only-defaults',slug);old_final=" stop();stage(w/'before-libgarnet.dylib');start('restored-control');print('RESTORED ORIGINAL CONTROL AND LIVE ASSIGNMENTS',flush=True)";assert old_final in t;t=t.replace(old_final," stop();stage(w/'before-libgarnet.dylib');subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True);print('RESTORED VALIDATED WS10 AND NORMAL NONBENCHMARK SERVER',flush=True)");p.write_text(t)
t=(old/'prepare_cohort.py').read_text().replace('worldsense-isolated-primary',slug);start=t.index('assert all(v==0');end=t.index("subprocess.run(['python3',str(w/'freeze.py')],check=True)");replacement="assert json.loads((w/'admission-complete.json').read_text())['ten_expected_config_rejections']\nassert json.loads((w/'correctness-terminal.json').read_text())['controller_session']==31658\nassert json.loads((w/'full-proof-complete.json').read_text())['all_original_native_model_vision_session_and_QoS_typed_scope_parent_death_guards_pass']\nassert json.loads((w/'post-owner-death-recovery.json').read_text())['exact_28layer_native_state_tokens_text_prompt_profile_after_active_owner_death_pass']\nassert json.loads((w/'acceptance-gate-audit.json').read_text())['all_five_original_golden_numeric_metric_gate_AST_unchanged']\n";t=t[:start]+replacement+t[end:];(w/'prepare_cohort.py').write_text(t)
t=(old/'run_resource.py').read_text();needle=" subprocess.run([sys.executable,str(w/'analyze_resource.py')],check=True)";assert needle in t;t=t.replace(needle," subprocess.run([sys.executable,str(w/'audit_native_cohort.py')],check=True)\n"+needle);(w/'run_resource.py').write_text(t)
# Numeric/golden/input/threshold logic must be exactly the established AST.
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
checks={}
for rel in analyzers:
 a=ast.dump(Normalize().visit(ast.parse((old/rel).read_text())),include_attributes=False);b=ast.dump(Normalize().visit(ast.parse((w/rel).read_text().replace(slug,'worldsense-isolated-primary'))),include_attributes=False);assert a==b,rel;checks[rel]=True
for name in ['freeze.py','dependency_closure.py','prepare_cohort.py','run_resource.py','controller.py']+analyzers:ast.parse((w/name).read_text())
(w/'acceptance-gate-audit.json').write_text(json.dumps(dict(all_five_original_golden_numeric_metric_gate_AST_unchanged=True,files=checks,only_native_architecture_admission_replaced=True,full_route_mutation_audit_pending=True,driver_sha256={rel:hashlib.sha256((w/rel).read_bytes()).hexdigest()for rel in analyzers}),indent=2)+'\n')
print('Prepared WorldSense drivers: original five golden/metric/threshold ASTs exact; native admission mutation audit pending; no dispatch')
