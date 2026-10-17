from pathlib import Path
import ast,json,sys,hashlib
w=Path(__file__).resolve().parent;label=sys.argv[1];enabled=label.startswith('candidate');load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();d=load(w/(label+'.json'));env=load(w/'environment.json');assert d['valid']and d['assignments_restored']and d['executor']=='XLang3'and len(d['items'])==5 and not d['search_plans']and d['request']==load(w/'cases.json')
assert [(i['repetition'],i['warmup'],i['policy'])for i in d['items']]==[(i,i==0,'default')for i in range(5)]
fn=next(n for n in ast.parse((w/'analyze.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(w/'analyze.py'),'exec'),ns)
gold=load(Path('/Users/shawn/CantorAI/Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json'));refs={(i['observation_id'],phase):(i,c)for i in gold['items']if i['policy']=='candidate'for phase,c in i['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'};count=0
for i in d['items']:
 for phase,c in i['result']['model_response'].items():
  if not isinstance(c,dict)or'model_id'not in c:continue
  assert c['status']=='ok';refi,ref=refs[(i['observation_id'],phase)];stable=lambda item:{k:v for k,v in item['result'].items()if k not in ['model_response','model_response_json']};assert stable(i)==stable(refi)
  assert all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])and c['timings']['input_tokens']==ref['timings']['input_tokens'];assert all(i[k]==refi[k]for k in ['image_sha256','skill_sha256'])
  assert ns['native_route'](c,enabled);assert Path(c['timings']['serving_source']).resolve()==Path(env['new_source'if enabled else'old_source'])/'runtime/metal_serving.py';count+=1
assert count==10
worker=next(x for x in load(w/'cohort-manifest.json')if x['label']==label);assert worker['client_exit']==0 and worker['library_sha256']==env['libraries']['on'if enabled else'off']and worker['serving_sha256']==sha(Path(env['new_source'if enabled else'old_source'])/'runtime/metal_serving.py')
(w/('validation-'+label+'.json')).write_text(json.dumps({'worker':label,'calls':10,'all_original_outputs_inputs_configs_semantics_exact':True,'raw_native_policy_valid':True,'library_source_identity_valid':True,'data_sha256':sha(w/(label+'.json'))},indent=2)+'\n');print('Validated10 raw calls/output/native/library/source',label,flush=True)
