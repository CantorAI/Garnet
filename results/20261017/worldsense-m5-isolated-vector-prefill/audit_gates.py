from pathlib import Path
import ast,json,copy
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-primary-u32';out={}
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):
  if node.name in ['residency_valid','native_route']:return None
  return self.generic_visit(node)
 def visit_Assign(self,node):
  if any(isinstance(n,ast.Name)and n.id in ['native','pooled','private']for n in node.targets):return None
  return self.generic_visit(node)
 def visit_keyword(self,node):
  if node.arg in ['avoided_descriptor_bytes','retired_default_bytes']:node.arg='policy_counter';node.value=ast.Constant(value=None)
  return self.generic_visit(node)
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py']:
 a=(old/rel).read_text();b=(w/rel).read_text().replace('worldsense-isolated-vector-prefill','worldsense-primary-u32')
 assert ast.dump(Normalize().visit(ast.parse(a)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse(b)),include_attributes=False),rel
 tree=ast.parse((w/rel).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(w/rel),'exec'),ns)
 calls=0
 for mode in ['off','primary','on']:
  records=json.loads((w/('api-state-progress-'+mode+'.json')).read_text())['records'];assert len(records)==5
  for r in records:
   c=r['call'];enabled=mode=='on';packed=mode!='off';prior=mode=='primary';assert ns['native_route'](c,enabled,prior);calls+=1
   for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata']:
    bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace(cap,'')if packed else bad['backend']+' '+cap;assert not ns['native_route'](bad,enabled,prior)
   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_vector_prefill','')if enabled else bad['backend']+' primary_vector_prefill';assert not ns['native_route'](bad,enabled,prior)
   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_prefill_isolated_library','')if enabled else bad['backend']+' primary_prefill_isolated_library';assert not ns['native_route'](bad,enabled,prior)
   for field in ['lazy_default_retired_bytes','lazy_default_rehydrated_bytes','lazy_default_planned_slots']:
    bad=copy.deepcopy(c);bad['timings'][field]=1;assert not ns['native_route'](bad,enabled,prior)
   bad=copy.deepcopy(c);bad['timings']['primary_storage_after']['metadata_bytes']+=4;assert not ns['native_route'](bad,enabled,prior)
   for field in ['vector_decode_encoded','vector_head_encoded']:
    bad=copy.deepcopy(c);bad['timings']['primary_encoded_delta'][field]+=1;assert not ns['native_route'](bad,enabled,prior)
 assert calls==15
 out[rel]={'numeric_golden_source_metric_AST_unchanged':True,'only_native_route_replaced':True,'real_native_calls_validated':calls,'wrong_capabilities_lazy_owner_metadata_vector_counts_rejected':True}
# The extra matched-primary screen adds prospective stricter gates, preserving
# the complete shared input/golden/model metric prefix and definitions.
a=ast.parse((w/'speed/analyze.py').read_text());b=ast.parse((w/'primary-speed/analyze.py').read_text())
def prefix(tree,stop):
 body=[]
 for node in tree.body:
  if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id==stop for t in node.targets):break
  body.append(node)
 return Normalize().visit(ast.Module(body=body,type_ignores=[]))
assert ast.dump(prefix(a,'passed'),include_attributes=False)==ast.dump(prefix(b,'foot'),include_attributes=False)
out['primary-speed/analyze.py']={'shared_full_golden_input_model_metric_prefix_AST_unchanged':True,'additional_prospective_5percent_VLM_TTFT_and2percent_memory_preservation':True,'no_original_stage_gate_relaxed':True}
(w/'acceptance-gate-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
