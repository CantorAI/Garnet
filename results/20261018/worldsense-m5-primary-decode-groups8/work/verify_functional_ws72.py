"""Independently verify four real-pipeline functional calls, not performance."""
from pathlib import Path
import json,hashlib,ast,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-primary-decode-groups8';p=w/'functional-smoke';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-decode-groups8/source'
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
identity=json.loads((p/'before-functional-identities.json').read_text())
assert all(sha(s/k)==v for k,v in identity['source'].items())and all(sha(Path(k))==v for k,v in identity['drivers'].items())
assert all(sha(r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role)/'libgarnet.dylib')==v for role,v in identity['libraries'].items())
tree=ast.parse((w/'analyze_resource.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'native route','exec'),ns)
data=json.loads((p/'candidate-1.json').read_text());assert data['valid']and data['assignments_restored']and len(data['items'])==2 and not data['search_plans']and data['request']==json.loads((p/'cases.json').read_text())
cohort=json.loads((p/'cohort-manifest.json').read_text());assert len(cohort)==1 and cohort[0]['label']=='candidate-1'and cohort[0]['library_sha256']==identity['libraries']['on']and cohort[0]['serving_sha256']==identity['source']['runtime/metal_serving.py']
gold=json.loads((r/'Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json').read_text());refs={(i['observation_id'],phase):(i,c)for i in gold['items']if i['policy']=='candidate'for phase,c in i['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'}
checks=[]
for j,item in enumerate(data['items']):
 assert(item['repetition'],item['warmup'],item['policy'])==(j,j==0,'default')
 for phase,c in item['result']['model_response'].items():
  if not isinstance(c,dict)or c.get('status')!='ok':continue
  ri,ref=refs[(item['observation_id'],phase)]
  stable=lambda x:{k:v for k,v in x.items()if k not in ['model_response','model_response_json']}
  assert stable(item['result'])==stable(ri['result'])
  assert all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
  assert c['timings']['input_tokens']==ref['timings']['input_tokens']and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256']
  assert Path(c['timings']['serving_source']).resolve()==s/'runtime/metal_serving.py'
  assert ns['native_route'](c,True)
  memory=c['timings']['memory_after_decode'];assert memory['task_info_status']==0
  for key in ['metal_allocated_bytes','task_physical_footprint_bytes','task_resident_bytes','task_compressed_bytes','task_internal_bytes','task_virtual_bytes']:assert isinstance(memory[key],int)and memory[key]>=0
  if c['model_id']=='Qwen3.5-4B':assert c['image_preprocess_backend']=='native-cpp'
  checks.append(dict(model=c['model_id'],phase=phase,exact=True,core_deltas={key:c['timings']['primary_storage_after'][key]-c['timings']['primary_storage_before'][key]for key in ['core_library_decode_encoded','core_library_prefill_encoded','core_library_head_encoded']},primary_storage=c['timings']['primary_storage_after'],primary_encoded_delta=c['timings']['primary_encoded_delta']))
assert len(checks)==4
assert json.loads((w/'restoration-verification.json').read_text())['benchmark_valid_request_http404']
(p/'functional-verification.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actualWorldSense_Galaxy_native_candidate_functional_pass=True,four_calls_exact=True,actual_groups8_deltas_match_primary_core_zero_and_native_memory_pass=True,checks=checks,source_drivers_libraries_unchanged=True,speed_eligibility=False,scope='Two fixed-frame candidate-only Skill requests. No control, speed comparison, causal attribution or promotion.'),indent=2)+'\n')
print('Actual WorldSense/Galaxy four-call golden/native/groups8/source/memory correctness PASS; no speed claim')
