"""Independent exact-golden, route, scope and complete paired counter audit."""
from pathlib import Path
import ast,datetime,hashlib,json,math,re
b=Path(__file__).resolve().parent.parent;w=b/'worldsense-paired-gpu-profile';p=w/'capture';r=Path('/Users/shawn/CantorAI')
load=lambda q:json.loads(q.read_text())
def sha(q):
 h=hashlib.sha256()
 with q.open('rb')as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
frozen=load(p/'identities-before-dispatch.json');assert all(sha(Path(k))==v for k,v in frozen['external'].items())
for family,m in [('current',frozen['source']),('legacy',frozen['legacy_source'])]:
 root=r/('out/research/worldsense-paired-gpu-profile-'+family)/'source';assert all(sha(root/k)==v for k,v in m.items())
assert load(p/'dependencies-after-actual-capture.json')['all_aliases_and_file_bytes_unchanged']
terminal=load(p/'controller-terminal.json');assert terminal['success']and terminal['normal_restored']and not terminal['promotion']and not terminal['speed_eligibility']
cohort=load(p/'cohort-manifest.json');assert [x['label']for x in cohort]==['legacy-1','current-1','current-2','legacy-2']and len({x['pid']for x in cohort})==4
assert all(x['client_exit']==0 and x['owned_worker_stopped']for x in cohort)
gold=load(r/'Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json')
refs={(item['observation_id'],phase):(item,c)for item in gold['items']if item['policy']=='candidate'for phase,c in item['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'}
ns={};exec(compile(ast.parse((p/'native_route_admission.py').read_text()),'original74 native admission','exec'),ns)
seed=b/'worldsense-minimal-selective-defaults/production-speed/analyze.py';fn=next(x for x in ast.parse(seed.read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='native_route')
assert (p/'native_route_admission.py').read_text()==ast.get_source_segment(seed.read_text(),fn)+'\n'
requests={};call_rows=[];command_rows=[];normalized_sequences={};all_intervals=0
for worker,state in zip(frozen['workers'],cohort):
 label=worker['label'];family=worker['family'];pid=state['pid'];assert all(state[k]==worker[k]for k in worker)
 data=load(p/(label+'.json'));assert data['valid']and data['assignments_restored']and len(data['items'])==2 and not data['search_plans']and data['request']==load(p/'cases.json')
 expected={};request_map={}
 for j,item in enumerate(data['items']):
  assert(item['repetition'],item['warmup'],item['policy'])==(j,j==0,'default');models=set()
  for phase,call in item['result']['model_response'].items():
   if not isinstance(call,dict)or call.get('status')!='ok':continue
   ref_item,ref=refs[(item['observation_id'],phase)];stable=lambda obj:{k:v for k,v in obj.items()if k not in ['model_response','model_response_json']}
   assert stable(item['result'])==stable(ref_item['result'])and all(call[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
   assert item['image_sha256']==ref_item['image_sha256']and item['skill_sha256']==ref_item['skill_sha256']and call['timings']['input_tokens']==ref['timings']['input_tokens']
   assert str(Path(call['timings']['serving_source']).resolve())==worker['serving_source']and 'gpu_pass_counter_profile'in call['backend']and 'native_gateway'not in call['timings']
   assert ns['native_route'](call,family=='current')
   if family=='current':assert call['timings']['memory_after_decode']['task_info_status']==0
   else:assert not any(key in call['timings']for key in ['primary_storage_before','primary_storage_after','primary_encoded_delta','memory_after_decode'])
   if call['model_id']=='Qwen3.5-4B':assert call['image_preprocess_backend']=='native-cpp'
   assert call['generated_tokens']==len(call['generated_token_ids'])
   for key in ['ttft_seconds','total_seconds']:assert type(call[key])in [int,float]and math.isfinite(call[key])and call[key]>0
   assert call['total_seconds']>=call['ttft_seconds']
   request=call['timings']['counter_trace_request_id'];assert type(request)is int and request>0 and request not in request_map
   count=call['generated_tokens'];limit=call['serving_configuration']['requested_output_limit'];steps=count-int(count==limit);sampled=[v for v in [4,16,28]if v<=steps];assert sampled==[4,16,28]
   row=dict(label=label,pid=pid,request=request,family=family,model=call['model_id'],phase=phase,repetition=j,warmup=item['warmup'],decode_steps=steps,expected_sampled_steps=sampled,exact=True)
   request_map[request]=row;call_rows.append(row);models.add(call['model_id'])
   for step in sampled:expected[(request,step)]=row
  assert models=={'Qwen3.5-4B','Qwen3-1.7B'}
 assert len(request_map)==4
 requests[label]=request_map;pending=[];seen=set();last={};samples=0
 trace=p/(label+'.counters.jsonl');assert sha(trace)==sha(Path(worker['trace']))
 for line in trace.read_text().splitlines():
  kind,payload=line.split(' ',1);v=json.loads(payload)
  assert v['pid']==pid and v['request']in request_map and v['model']==request_map[v['request']]['model']and (v['request'],v['decode_step'])in expected
  assert v['diagnostic']is True and v['serialized']is False
  if kind=='metal_counter_profile':
   assert v['index']==len(pending)and 0<v['start_raw']<=v['end_raw']<2**64-1;pending.append(v);samples+=1
  else:
   assert kind=='metal_counter_command'and v['samples']==len(pending)>0 and v['samples']<=2048 and v['partial_submissions']==0 and type(v['gpu_span_ms'])in [int,float]and math.isfinite(v['gpu_span_ms'])and v['gpu_span_ms']>=0
   assert len(pending)==({'Qwen3.5-4B':749,'Qwen3-1.7B':537}[v['model']])
   assert all(all(row[k]==v[k]for k in ['pid','context','command','request','model','decode_step'])for row in pending)
   key=(v['request'],v['decode_step']);assert key not in seen;seen.add(key);assert v['command']>last.get(v['context'],0);last[v['context']]=v['command']
   normalized=[row['op'][:-8]+':raw'if row['op'].endswith(':primary')else row['op']for row in pending]
   sequence_key=(v['model'],v['decode_step'])
   if sequence_key in normalized_sequences:assert normalized==normalized_sequences[sequence_key]
   else:normalized_sequences[sequence_key]=normalized
   command_rows.append(dict(label=label,**v,raw_trace_sha256=sha(trace),ordered_normalized_ops_sha256=hashlib.sha256(json.dumps(normalized,separators=(',',':')).encode()).hexdigest()));pending=[]
 assert not pending and seen==set(expected)and len(seen)==12;all_intervals+=samples
assert len(call_rows)==16 and len(command_rows)==48 and all_intervals==30864
assert all(load(w/'restoration-verification.json')[key]for key in ['validated_WS10_binary_restored','four_file_sources_online','camera_unmounted','benchmark_valid_request_http404'])
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sixteen_actual_WorldSense_Galaxy_native_calls_exact=True,eight_Skill_requests=True,four_fresh_owned_workers_mirrored=True,counter_commands=len(command_rows),counter_intervals=all_intervals,all48_reached_steps4_16_28_exact_scope_and_PID_join=True,all_aligned_ordered_operation_sequences_equal_after_declared_primary_raw_label_normalization=True,original74_native_admission_and_all_original_goldens_preserved=True,all_source_model_runtime_library_driver_and_host_identities_unchanged=True,normal_WS10_fourfiles_noCam_benchmark404_restored=True,speed_eligibility=False,promotion=False,calls=call_rows,commands=command_rows,scope='Paired reconstructed-source observer diagnostic. Raw timestamps retain raw units; source/whole-profile differences and observer effects prevent isolated cause or throughput acceptance claims.')
(p/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('WS76 paired16actualWorldSense calls/48native counter windows/30864intervals exact; no speed/promotion claim')
