from pathlib import Path
import ast,json,copy
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-primary-vector';out={}
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):
  if node.name in ['residency_valid','native_route']:return None
  return self.generic_visit(node)
 def visit_Assign(self,node):
  if any(isinstance(n,ast.Name)and n.id in ['native','pooled','private']for n in node.targets):return None
  return self.generic_visit(node)
 def visit_keyword(self,node):
  if node.arg in ['avoided_descriptor_bytes','retired_default_bytes']:
   node.arg='policy_counter';node.value=ast.Constant(value=None)
  return self.generic_visit(node)
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py']:
 a=(old/rel).read_text();b=(w/rel).read_text().replace('worldsense-primary-kmajor','worldsense-primary-vector')
 assert ast.dump(Normalize().visit(ast.parse(a)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse(b)),include_attributes=False),rel
 tree=ast.parse((w/rel).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');namespace={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(w/rel),'exec'),namespace)
 calls=0
 for mode in ['off','on']:
  d=json.loads((w/('api-state-progress-'+mode+'.json')).read_text())
  assert len(d['records'])==5
  for r in d['records']:
   c=r['call'];assert namespace['native_route'](c,mode=='on');calls+=1
   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_bf16_blocks','')if mode=='on'else bad['backend']+' primary_bf16_blocks';assert not namespace['native_route'](bad,mode=='on')
   bad=copy.deepcopy(c);bad['timings']['lazy_default_rehydrated_bytes']=1;assert not namespace['native_route'](bad,mode=='on')
   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_vector_decode','')if mode=='on'else bad['backend']+' primary_vector_decode';assert not namespace['native_route'](bad,mode=='on')
   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_kmajor_decode','')if mode=='on'else bad['backend']+' primary_kmajor_decode';assert not namespace['native_route'](bad,mode=='on')
   bad=copy.deepcopy(c);bad['timings']['primary_encoded_delta']['vector_decode_encoded']+=1;assert not namespace['native_route'](bad,mode=='on')
   bad=copy.deepcopy(c);bad['timings']['primary_encoded_delta']['vector_head_encoded']+=1;assert not namespace['native_route'](bad,mode=='on')

 assert calls==10
 out[rel]={'numeric_golden_source_metric_AST_unchanged':True,'only_native_route_and_counter_column_replaced':True,'real_native_correctness_calls_validated':calls,'wrong_primary_vector_Kmajor_capability_rehydration_or_vector_route_counts_rejected':True}
(w/'acceptance-gate-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
