from pathlib import Path
import base64,collections,hashlib,json,sys,zipfile
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent;repo=Path(sys.argv[2]) if len(sys.argv)>2 else Path('D:/CantorAI/work/qwen38-5090');sys.path.insert(0,str(repo/'test2026/cuda/qwen38_5090'))
from compare_http_runs import verify_run
archive=out/'native-4b-http-evidence.zip'
with zipfile.ZipFile(archive) as z,zipfile.ZipFile(out/'qwen35-4b-grid16-evidence.zip') as old,zipfile.ZipFile(out/'native-mtp3-shared-joined-evidence.zip') as big:
 assert z.testzip() is None;e=json.loads(z.read('evidence-manifest.json'))
 for n,h in e['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
 for n,h in e['linked_archives'].items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h,n
 base='results/native-4b-http-pilot/';q=json.loads(z.read(base+'quality.json'));assert q['status']=='passed'
 old_report=json.loads(old.read('results/qwen35-4b-grid16-smoke/report.json'));config=z.read('models/Qwen3.5-4B/config.json')
 expected_config=next(h for n,h in old_report['files'].items() if n.replace('\\','/').endswith('/models/Qwen3.5-4B/config.json'));assert hashlib.sha256(config).hexdigest()==expected_config
 for case in ('text','image'):
  teacher=next(row for row in old_report['cases'] if row['key']==case);result=teacher['result'];ids=list(result['generated_token_ids'][0])
  if result['stop_token_ids'][0] is not None:ids.append(result['stop_token_ids'][0])
  for repeat in range(2):
   stem=base+'old-'+case+'-'+str(repeat);body=json.loads(z.read(stem+'-request.json'));response=json.loads(z.read(stem+'-response.json'))['choices'][0]
   assert body['model']=='qwen35-4b-bf16' and body['max_tokens']==16 and response['prompt_token_ids']==result['prompt_token_ids'][0] and response['response_token_ids']==ids
   if case=='image':assert base64.b64decode(body['messages'][0]['content'][0]['image_url']['url'].split(',',1)[1])==old.read('qwen35-4b-smoke.jpg')
 for key,want in [('red','red'),('green','green'),('blue','blue'),('bottom-blue','blue')]:
  response=json.loads(z.read(base+'semantic-'+key+'-response.json'));assert response['choices'][0]['message']['content'].strip().lower().rstrip('.')==want
 records={}
 for case in ('text','image'):
  for i in range(8):
   key='s3-'+case+'-c1-r'+str(i);rows=[]
   for repeat in range(2):
    stem=base+key+'-repeat-'+str(repeat);rows.append(json.loads(z.read(stem+'-response.json'))['choices'][0])
   assert z.read(base+key+'-repeat-0-request.json')==z.read(base+key+'-repeat-1-request.json')
   assert all(rows[0][f]==rows[1][f] for f in ('prompt_token_ids','response_token_ids')) and len(rows[0]['response_token_ids'])==128 and len(rows[0]['prompt_token_ids'])==(41 if case=='text' else 104);records[key]=rows[0]
 def final(n):
  events=[json.loads(l)['data'] for l in z.read(n).decode().splitlines()];assert events[-1]=='[DONE]';return next(json.loads(v) for v in reversed(events) if v!='[DONE]')
 for case in ('text','image'):
  stream=final(base+'stream-gate-'+case+'-events.jsonl');assert all(stream[f]==records['s3-'+case+'-c1-r0'][f] for f in ('prompt_token_ids','response_token_ids'))
 names=[n for n in z.namelist() if n.startswith(base+'benchmark/') and n.endswith('-manifest.json')];assert len(names)==1
 manifest=json.loads(z.read(names[0]));prefix=names[0].rsplit('/',1)[0]+'/';run=verify_run(manifest,lambda n:z.read(prefix+n));assert len(run['cohorts'])==2
 workers={}
 for line in z.read('results/native-4b-http-worker/worker-stdout.jsonl').decode().splitlines():
  try:event=json.loads(json.loads(line)['line'])
  except ValueError:continue
  if isinstance(event,dict) and event.get('event')=='completed':assert event['id'] not in workers;workers[event['id']]=event
 old_manifest_name=next(n for n in big.namelist() if '/benchmark/' in n and n.endswith('-manifest.json'));old_manifest=json.loads(big.read(old_manifest_name));old_prefix=old_manifest_name.rsplit('/',1)[0]+'/'
 samples=[json.loads(l) for l in z.read(prefix+manifest['run_id']+'-vram.jsonl').decode().splitlines()];measured=0;prompt_matches=0
 for cohort in manifest['cohorts']:
  assert cohort['concurrency']==1 and len(cohort['request_keys'])==8
  rows=[r for r in samples if r['phase']==cohort['case']+'-c1'][:cohort['samples']];assert len(rows)==cohort['samples'] and max(r['used_bytes'] for r in rows)==cohort['sampled_peak_adapter_used_bytes']
  for key in cohort['request_keys']:
   event=final(prefix+key+'-events.jsonl');work=workers[event['id']];workload=key.removeprefix(manifest['run_id']+'-');expected=records[workload]
   assert event['prompt_token_ids']==work['prompt_ids']==expected['prompt_token_ids'] and event['response_token_ids']==work['generated_ids']==expected['response_token_ids']
   assert event['study']['generation_mode']==work['generation_mode']=='greedy_target' and event['study']['speculative_metrics'] is None
   body=json.loads(z.read(prefix+key+'-request.json'));old_key=old_manifest['run_id']+'-'+workload;old_body=json.loads(big.read(old_prefix+old_key+'-request.json'));body.pop('model');old_body.pop('model');assert body==old_body
   old_events=[json.loads(l)['data'] for l in big.read(old_prefix+old_key+'-events.jsonl').decode().splitlines()];old_final=next(json.loads(v) for v in reversed(old_events) if v!='[DONE]');prompt_matches+=event['prompt_token_ids']==old_final['prompt_token_ids']
   if cohort['case']=='image':assert hashlib.sha256(base64.b64decode(body['messages'][0]['content'][0]['image_url']['url'].split(',',1)[1])).hexdigest()==work['image_sha256']
   measured+=1
 assert measured==16
 metadata=json.loads(z.read(base+'metadata.json'));assert metadata['health']['native_worker']['model']=='qwen35-4b-bf16'
 for name,h in metadata['file_sha256'].items():
  relative=name.replace('\\','/').split('5090-study-20261023/',1)[1]
  if relative.startswith('build/bin/'):continue
  if relative.endswith('.zip'):assert hashlib.sha256((out/Path(relative).name).read_bytes()).hexdigest()==h
  else:assert hashlib.sha256(z.read(relative)).hexdigest()==h,relative
 for folder in ('native-4b-http-quality-ledger','native-4b-http-benchmark-ledger'):
  events=[json.loads(l) for l in z.read('results/'+folder+'/commands.jsonl').decode().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
 stop=json.loads(z.read('results/native-4b-http-intentional-stop.json'));assert stop['taskkill_exit']==0 and stop['quality_exit_code']==stop['benchmark_exit_code']==0
 resource=json.loads(z.read('results/native-4b-http-server-resources/resource-monitor.json'));all_samples=[json.loads(l) for l in z.read('results/native-4b-http-server-resources/resource-vram.jsonl').decode().splitlines()];assert len(all_samples)==resource['sampled_memory']['samples'] and max(v['used_bytes'] for v in all_samples)==resource['sampled_memory']['sampled_peak_adapter_used_bytes']
 result={'status':'native4b_http_evidence_verified','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files_verified':len(e['files']),'library_sha256':e['library_sha256'],'old_smoke_captures':4,'vlm_semantic_cases':4,'full128_repeat_pairs':16,'real_sse_gates':2,'measured_sse_worker_and_greedy_ids':measured,'requests_match_27b_except_model_alias':16,'prompt_ids_match_27b':prompt_matches,'run':run,'raw_cohorts':manifest['cohorts'],'whole_lifetime_resource':resource,'two_x_established':False,'scope':'One4B BF16 target-only B1/T16/cap4096 short-prompt pilot after retained/repeat/SSE gates. Different architecture/precision/speculation and binary from27B; descriptive comparison, not causal model-size/precision gain, broad quality parity, repeated confidence,4B reference, GPU batching/full-context or2x admission. Memory is sampled adapter-wide, not instantaneous process peak.'}
(out/'native-4b-http-verified.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ('run','whole_lifetime_resource')},indent=2))
