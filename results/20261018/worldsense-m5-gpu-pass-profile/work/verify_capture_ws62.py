"""Four actual WorldSense golden calls, exact counter identities, all frozen inputs."""
from pathlib import Path
import json,hashlib,re,datetime,ast
base=Path(__file__).resolve().parent.parent;w=base/'worldsense-gpu-counter-profile';p=w/'capture';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-gpu-counter-profile/source'
def sha(q):
 h=hashlib.sha256()
 with q.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
frozen=json.loads((p/'identities-before-dispatch.json').read_text());assert all(sha(s/k)==v for k,v in frozen['source'].items())and all(sha(Path(k))==v for k,v in frozen['external'].items())
assert json.loads((w/'dependencies-after-actual-counter-capture.json').read_text())['all_aliases_and_file_bytes_unchanged']
data=json.loads((p/'candidate-1.json').read_text());assert data['valid']and data['assignments_restored']and len(data['items'])==2 and not data['search_plans']and data['request']==json.loads((p/'cases.json').read_text())
cohort=json.loads((p/'cohort-manifest.json').read_text());assert len(cohort)==1 and cohort[0]['client_exit']==0
pid=int(re.search(r'\n\s+pid = (\d+)',cohort[0]['cantor_before']).group(1));assert cohort[0]['library_sha256']==frozen['external'][str(r/'out/build/garnet-metal-worldsense-gpu-counter-profile-on/libgarnet.dylib')]
assert cohort[0]['serving_sha256']==frozen['source']['runtime/metal_serving.py']
gold=json.loads((r/'Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json').read_text());refs={(item['observation_id'],phase):(item,c)for item in gold['items']if item['policy']=='candidate'for phase,c in item['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'}
route={};exec(compile(ast.parse((p/'native_route_admission.py').read_text()),'unchanged native route admission','exec'),route)
requests={};checks=[]
for j,item in enumerate(data['items']):
 assert(item['repetition'],item['warmup'],item['policy'])==(j,j==0,'default')
 for phase,c in item['result']['model_response'].items():
  if not isinstance(c,dict)or c.get('status')!='ok':continue
  ref_item,ref=refs[(item['observation_id'],phase)];stable=lambda x:{k:v for k,v in x.items()if k not in ['model_response','model_response_json']}
  assert stable(item['result'])==stable(ref_item['result'])and all(c[key]==ref[key]for key in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
  assert item['image_sha256']==ref_item['image_sha256']and item['skill_sha256']==ref_item['skill_sha256']and c['timings']['input_tokens']==ref['timings']['input_tokens']
  assert Path(c['timings']['serving_source']).resolve()==s/'runtime/metal_serving.py'and 'gpu_pass_counter_profile'in c['backend']and 'native_gateway'not in c['timings']
  assert route['native_route'](c,True)
  if c['model_id']=='Qwen3.5-4B':assert c['image_preprocess_backend']=='native-cpp'
  assert c['timings']['memory_after_decode']['task_info_status']==0
  req=c['timings']['counter_trace_request_id'];assert isinstance(req,int)and not isinstance(req,bool)and req>0 and req not in requests
  count=c['generated_tokens'];limit=c['serving_configuration']['requested_output_limit'];steps=count-1 if count==limit else count
  requests[req]=dict(model=c['model_id'],phase=phase,warmup=item['warmup'],repetition=j,decode_steps=steps,expected_sampled_steps=[v for v in [4,16,28]if v<=steps])
  checks.append(dict(request=req,model=c['model_id'],exact=True))
assert len(requests)==4
samples=[];commands=[];pending=[];seen=set();last={}
for line in(p/'candidate-1.counters.jsonl').read_text().splitlines():
 kind,payload=line.split(' ',1);c=json.loads(payload);assert c['pid']==pid and c['request']in requests and c['model']==requests[c['request']]['model']and c['decode_step']in requests[c['request']]['expected_sampled_steps']
 assert c['diagnostic']and not c['serialized']
 if kind=='metal_counter_profile':
  assert c['index']==len(pending)and 0<c['start_raw']<=c['end_raw']<2**64-1;pending.append(c);samples.append(c)
 else:
  assert kind=='metal_counter_command'and c['samples']==len(pending)>0 and c['samples']<=2048 and c['partial_submissions']==0
  assert all(all(row[key]==c[key]for key in ['pid','context','command','request','model','decode_step'])for row in pending)
  key=(c['request'],c['decode_step']);assert key not in seen;seen.add(key)
  assert c['command']>last.get(c['context'],0);last[c['context']]=c['command'];commands.append(c);pending=[]
assert not pending and seen=={(req,step)for req,data in requests.items()for step in data['expected_sampled_steps']}
assert sha(Path(frozen['capture_trace_path']))==sha(p/'candidate-1.counters.jsonl')
assert json.loads((w/'restoration-verification.json').read_text())['benchmark_valid_request_http404']
(p/'verification.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),four_actualWorldSense_Galaxy_native_calls_exact=True,pid=pid,requests=requests,counter_commands=len(commands),counter_samples=len(samples),all_reached_predeclared_steps_and_native_scopes_join_exactly=True,all_source_model_runtime_library_driver_and_host_identities_unchanged=True,WS10_normal_fourfiles_noCam_benchmark404_restored=True,speed_eligibility=False,scope='Candidate-only bounded hardware-counter diagnostic. No throughput gain, complete GPU decomposition or promotion.'),indent=2)+'\n')
print('WS62 actualWorldSense4calls exact; all declared hardware-counter request/model/step identities and frozen inputs PASS')
