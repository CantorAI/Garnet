from pathlib import Path
import ast,json
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-private-resident-defaults';out={}
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py']:
 text=(w/rel).read_text();tree=ast.parse(text);functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='residency_valid'];assert len(functions)==1;function=functions[0];namespace={};exec(compile(ast.Module(body=[function],type_ignores=[]),str(w/rel),'exec'),namespace);validator=namespace['residency_valid'];calls=0
 for mode in ['off','on']:
  d=json.loads((w/('api-state-progress-'+mode+'.json')).read_text())
  for r in d['records']:
   t=r['call']['timings'];assert validator(t,mode=='on',r['model_id']);calls+=1
   bad=dict(t);bad['weight_residency']=dict(t['weight_residency'],active_bytes=1);assert not validator(bad,mode=='on',r['model_id'])
 lines=text.splitlines(True);text=''.join(lines[:function.lineno-1]+lines[function.end_lineno:]);text=text.replace('worldsense-request-residency','worldsense-private-resident-defaults')
 text=text.replace("native=native and residency_valid(t,mode=='on',c['model_id']);",'').replace("   native=native and residency_valid(timings,bool(expected),c['model_id'])\n",'').replace(" and residency_valid(timings,bool(expected),c['model_id'])",'')
 assert ast.dump(ast.parse(text),include_attributes=False)==ast.dump(ast.parse((old/rel).read_text()),include_attributes=False),rel
 out[rel]={'numeric_gate_and_metric_AST_unchanged':True,'real_native_correctness_calls_validated':calls,'nonzero_active_bytes_rejected':True,'scope':'Only sourcepath andnew residencyroutevalidationadded. Not a speed result.'}
(w/'acceptance-gate-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
