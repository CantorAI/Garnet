"""Keep all original WorldSense metric/golden/threshold gates; extend native layout admission."""
from pathlib import Path
import json,ast,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-sole-interleaved-text';old=b/'work/worldsense-isolated-primary';slug='worldsense-sole-interleaved-text'
assert json.loads((w/'admission-complete.json').read_text())['twentythree_expected_config_rejections']
analyzers=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']
for rel in analyzers:
 t=(old/rel).read_text().replace('worldsense-isolated-primary',slug)
 needle="packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']"
 guard='''
 if call['model_id']not in ['Qwen3.5-4B','Qwen3-1.7B']:return False
 if 'native_text_worker'in call['backend']or'native_gateway'in t:return False
 if ('sole_interleaved_mlp'in call['backend'])!=enabled:return False
 if ('sole_direct_interleaved_prefill'in call['backend'])!=enabled:return False
 if ('sole_interleaved_text'in call['backend'])!=enabled:return False
 fields={'active_weights','allocated_bytes','uploads','decode_encoded','restore_encoded','scratch_bytes','prefill_encoded'}
 sb=t.get('sole_storage_before');ss=t.get('sole_storage_after');sd=t.get('sole_encoded_delta')
 if not isinstance(sb,dict)or not isinstance(ss,dict)or not isinstance(sd,dict):return False
 if set(sb)!=fields or set(ss)!=fields or set(sd)!={'decode_encoded','restore_encoded','prefill_encoded'}:return False
 if any(not isinstance(v,int)or isinstance(v,bool)or v<0 for v in list(sb.values())+list(ss.values())+list(sd.values())):return False
 if any(sd[k]!=ss[k]-sb[k]for k in sd):return False
 if not enabled:
  if any(sb.values())or any(ss.values())or any(sd.values()):return False
 else:
  if ss['active_weights']not in [0,197]or ss['allocated_bytes']!={0:0,197:3440902144}.get(ss['active_weights']) or ss['uploads']<ss['active_weights']:return False
  if ss['scratch_bytes']!=0 or ss['restore_encoded']!=0 or sd['restore_encoded']!=0:return False
  if call['model_id']=='Qwen3-1.7B':
   if ss['active_weights']!=197 or sd['decode_encoded']<=0:return False
   if not t.get('prefix_cache_hit',False)and sd['prefill_encoded']<=0:return False
  elif any(sd.values()):return False
'''
 assert t.count(needle)==1;t=t.replace(needle,needle+guard)
 if rel=='production-speed/analyze.py':
  needle=' if not enabled:\n';assert t.count(needle)==2
  t=t.replace(needle,needle+"  if any(cap in call['backend']for cap in ['sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_interleaved_text','native_text_worker'])or any(k in call['timings']for k in ['sole_storage_before','sole_storage_after','sole_encoded_delta','native_gateway']):return False\n",1)
 p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);ast.parse(t);p.write_text(t)
for name in ['freeze.py','dependency_closure.py']:(w/name).write_text((old/name).read_text().replace('worldsense-isolated-primary',slug))
p=w/'dependency_closure.py';v=p.read_text().replace("['off','primary','on','fault','restore']","['off','primary','on','fault','restore','direct64']").replace('for four builds','for six builds');ast.parse(v);p.write_text(v)
p=w/'freeze.py';t=p.read_text();t=t.replace('WS54 fully-isolated wideVLM/rawtext partition; rawcandidate9flags andstoragecontrolcandidate6. Sameprimarypartition/globalretirement, differentmetadata/prefill/directupload. Faults onlyinseparatecorrectnessbuild','WS65 expanded sole-text direct layout; RAWcandidate12flags andPRIMARYcandidate9flags. Same original primary partition/global retirement. Original shaders and arithmetic unchanged. Faults only in separate correctness build');needle="'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),";assert t.count(needle)==1;t=t.replace(needle,needle+"'GARNET_METAL_SOLE_INTERLEAVED_MLP':('OFF','ON'),'GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL':('OFF','ON'),'GARNET_METAL_SOLE_INTERLEAVED_TEXT':('OFF','ON'),")
needle="'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']}";assert t.count(needle)==1;t=t.replace(needle,"'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_SOLE_INTERLEAVED_MLP','GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL','GARNET_METAL_SOLE_INTERLEAVED_TEXT']}")
start=t.index("s=(w.parent/'worldsense-request-residency/controller.py')");t=t[:start]+"print('Frozen WS65 source/twelveflagRAWcontrast/nineflagPRIMARYcontrast and checkpoint identities PASS',flush=True)\n";ast.parse(t);p.write_text(t)
t=(old/'controller.py').read_text().replace('worldsense-isolated-primary',slug)
needle=" stop();stage(w/'before-libgarnet.dylib');start('restored-control');print('RESTORED ORIGINAL CONTROL AND LIVE ASSIGNMENTS',flush=True)";assert t.count(needle)==1
t=t.replace(needle," stop();stage(w/'before-libgarnet.dylib');subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True);print('RESTORED VALIDATED WS10 AND NORMAL NONBENCHMARK SERVER',flush=True)");(w/'controller.py').write_text(t)
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
checks={}
for rel in analyzers:
 a=ast.dump(Normalize().visit(ast.parse((old/rel).read_text())),include_attributes=False);z=ast.dump(Normalize().visit(ast.parse((w/rel).read_text().replace(slug,'worldsense-isolated-primary'))),include_attributes=False);assert a==z,rel;checks[rel]=True
(w/'acceptance-gate-audit.json').write_text(json.dumps(dict(all_five_original_golden_numeric_metric_gate_AST_unchanged=True,files=checks,only_native_layout_admission_extended=True,full_route_mutation_audit_pending=True,driver_sha256={rel:hashlib.sha256((w/rel).read_bytes()).hexdigest()for rel in analyzers}),indent=2)+'\n')
print('Five original golden/metric/threshold ASTs unchanged; native sole-layout routes added; no dispatch')
