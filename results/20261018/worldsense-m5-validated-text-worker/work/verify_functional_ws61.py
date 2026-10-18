"""Verify all four actual-pipeline calls and dual backend/process provenance."""
from pathlib import Path
import json, hashlib, ast, datetime
root = Path(__file__).resolve().parent.parent
w = root/'worldsense-validated-text-worker'; p=w/'functional-smoke';r=Path('/Users/shawn/CantorAI')
s=r/'out/research/worldsense-validated-text-worker/source'
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
identity=json.loads((p/'before-functional-identities.json').read_text())
assert all(sha(s/k)==v for k,v in identity['source'].items())
assert all(sha(r/'out/research/worldsense-prefix-cache/source'/k)==v for k,v in identity['legacy_source'].items())
assert all(sha(Path(k))==v for k,v in identity['drivers'].items())
assert all(sha(r/('out/build/garnet-metal-worldsense-validated-text-worker-'+role)/'libgarnet.dylib')==v for role,v in identity['libraries'].items())
tree=ast.parse((w/'analyze_resource.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'native route','exec'),ns)
data=json.loads((p/'candidate-1.json').read_text())
assert data['valid']and data['assignments_restored']and len(data['items'])==2 and not data['search_plans']and data['request']==json.loads((p/'cases.json').read_text())
cohort=json.loads((p/'cohort-manifest.json').read_text())
assert len(cohort)==1 and cohort[0]['label']=='candidate-1'and cohort[0]['library_sha256']==identity['libraries']['on']and cohort[0]['serving_sha256']==identity['source']['runtime/metal_serving.py']
gold=json.loads((r/'Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json').read_text())
refs={(i['observation_id'],phase):(i,c)for i in gold['items']if i['policy']=='candidate'for phase,c in i['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'}
checks=[];stable_child=None;remote_count=0
for j,item in enumerate(data['items']):
 assert(item['repetition'],item['warmup'],item['policy'])==(j,j==0,'default')
 for phase,c in item['result']['model_response'].items():
  if not isinstance(c,dict)or c.get('status')!='ok':continue
  ri,ref=refs[(item['observation_id'],phase)]
  stable=lambda x:{k:v for k,v in x.items()if k not in ['model_response','model_response_json']}
  assert stable(item['result'])==stable(ri['result'])
  assert all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
  assert c['timings']['input_tokens']==ref['timings']['input_tokens']and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256']
  remote=c['model_id']=='Qwen3-1.7B'
  assert Path(c['timings']['serving_source']).resolve()==s/('runtime/validated_ws10/metal_serving.py'if remote else'runtime/metal_serving.py')
  assert ns['native_route'](c,True)
  g=c['timings']['native_gateway'];child=g['child'];who=(g['parent_pid'],child['support']['pid'],child['birth_id'])
  assert stable_child in [None,who];stable_child=who
  if remote:
   remote_count+=1;assert child['infer_calls']==remote_count==c['native_worker_call_index']
  checks.append(dict(model=c['model_id'],phase=phase,exact=True,route=g['route']))
assert len(checks)==4 and remote_count==2
assert json.loads((w/'restoration-verification.json').read_text())['benchmark_valid_request_http404']
(p/'functional-verification.json').write_text(json.dumps(dict(
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    actualWorldSense_Galaxy_native_candidate_functional_pass=True,four_calls_exact=True,
    dual_backend_provenance_and_whole_process_native_memory_pass=True,
    stable_owned_child_two_exact_calls_no_hidden_retry=True,checks=checks,
    source_drivers_libraries_unchanged=True,speed_eligibility=False,
    scope='Two fixed-frame candidate-only Skill requests. No speed comparison or promotion.'),indent=2)+'\n')
print('WS61 actual WorldSense/Galaxy four calls: exact goldens, native dual backend/clock/process memory PASS')
