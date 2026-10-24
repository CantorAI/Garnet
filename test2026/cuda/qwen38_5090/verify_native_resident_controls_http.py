from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
out=Path(__file__).parent;repo=Path('D:/CantorAI/work/qwen38-5090');sys.path.insert(0,str(repo/'test2026/cuda/qwen38_5090'))
from compare_http_runs import verify_run,compare
from run_shared_mtp_candidate import verify_records,KEYS,GOLDEN_SHA256
from validate_native_mtp_http import validate_worker_profile
from benchmark_stream import TokenEvents
archive=out/'native-resident-controls-http-evidence.zip'
with zipfile.ZipFile(archive) as z,zipfile.ZipFile(out/'native-mtp3-shared-joined-evidence.zip') as old:
 assert z.testzip() is None;e=json.loads(z.read('evidence-manifest.json'))
 for n,h in e['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
 for n,h in e['linked_archives'].items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h,n
 assert e['runtime_sha256']['build/bin/garnet.dll']==e['library_sha256'] and e['runtime_sha256']['build/bin/xlang3.exe']=='2e2db12a2dbc1134a3a4c662c080b91d500b37af488661630d4d580e7684dc10'
 assert e['library_sha256']=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
 assert e['terminal']['native-resident-controls-http-controller-ledger']['exit_code']==0
 assert e['terminal']['native-resident-controls-http-server-ledger']['event']=='finished'
 base='results/native-resident-controls-profile-pilot/';diagnostic='results/shared-resident-controls-http-diagnostic/'
 meta=json.loads(z.read(base+'metadata.json'));worker=validate_worker_profile(meta['health'],3,4,'direct',4,128,True,False,True);assert worker['resident_request_controls'] is True
 proof=json.loads(z.read('results/warm-mtp-resident-controls-attempt1-verified.json'));assert proof['status']=='resident_controls_correctness_evidence_verified' and proof['archive_sha256']==e['linked_archives']['warm-mtp-resident-controls-attempt1-evidence.zip']
 raw=z.read('results/shared-decoder-full-qualification/report.json');assert hashlib.sha256(raw).hexdigest()==GOLDEN_SHA256
 records={key:[json.loads(z.read(diagnostic+key+'-'+mode+'-response.json')) for mode in ('target','mtp')] for key in KEYS};gates=verify_records(json.loads(raw),records)
 for key in KEYS:
  expected=json.loads(old.read('results/shared-joined-mlp-http-diagnostic/'+key+'-target-response.json'))['choices'][0]
  old_body=old.read('results/shared-joined-mlp-http-diagnostic/'+key+'-target-request.json');assert json.loads(z.read(diagnostic+key+'-target-request.json'))==json.loads(old_body)
  for response in records[key]:
   choice=response['choices'][0];assert choice['prompt_token_ids']==expected['prompt_token_ids'] and choice['response_token_ids']==expected['response_token_ids']
 # Confirm exact producer methods are current reviewed versions; no archived code execution.
 for name in ('native_worker_profile.py','native_stream_worker.py','benchmark_stream.py','compare_http_runs.py','metrics.py','probe_shared_mtp_sequences.py'):
  assert z.read('Garnet/test2026/cuda/qwen38_5090/'+name)==(repo/'test2026/cuda/qwen38_5090'/name).read_bytes(),name
 receipt=json.loads(z.read('results/resident-controls-http-source-install/receipt.json'));assert receipt['status']=='installed' and receipt['library_sha256']==e['library_sha256'] and receipt['package_sha256']==hashlib.sha256(z.read('resident-controls-http-install.zip')).hexdigest()
 for n,v in receipt['files'].items():
  assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256'],n
  if 'previous_sha256' in v:assert hashlib.sha256(z.read('results/resident-controls-http-source-install/backup/'+n)).hexdigest()==v['previous_sha256']
 manifests=[n for n in z.namelist() if n.startswith(base+'benchmark/') and n.endswith('-manifest.json')];assert len(manifests)==1
 m=json.loads(z.read(manifests[0]));prefix=manifests[0].rsplit('/',1)[0]+'/';run=verify_run(m,lambda n:z.read(prefix+n));assert len(m['cohorts'])==2
 samples=[json.loads(l) for l in z.read(prefix+m['run_id']+'-vram.jsonl').decode().splitlines()]
 for cohort in m['cohorts']:
  assert cohort['concurrency']==1 and len(cohort['request_keys'])==8
  rows=[row for row in samples if row['phase']==cohort['case']+'-c1'][:cohort['samples']];assert len(rows)==cohort['samples'] and max(row['used_bytes'] for row in rows)==cohort['sampled_peak_adapter_used_bytes']
  for key in cohort['request_keys']:
   events=[json.loads(l)['data'] for l in z.read(prefix+key+'-events.jsonl').decode().splitlines()];final=next(json.loads(v) for v in reversed(events) if v!='[DONE]');metrics=final['study']['speculative_metrics'];original=records[key.removeprefix(m['run_id']+'-')][0]['choices'][0]
   assert final['response_token_ids']==original['response_token_ids'] and final['prompt_token_ids']==original['prompt_token_ids']
   assert final['study']['generation_mode']=='verified_mtp' and metrics['target_head_reuses']==0
   assert metrics['target_head_evaluations']==metrics['target_verification_calls']+metrics['target_prefix_replays']
 for case in ('text','image'):
  stem=base+'stream-gate-'+case;events=[json.loads(l) for l in z.read(stem+'-events.jsonl').decode().splitlines()];saved=json.loads(z.read(stem+'-result.json'));body=json.loads(z.read(stem+'-request.json'));state=TokenEvents()
  for event in events:state.feed(event['data'],event['monotonic'])
  replay=state.result(saved['submitted'],saved['completed'],body['max_tokens'])
  assert replay['generated_tokens']==128 and replay['prompt_tokens']==(41 if case=='text' else 104)
  final=next(json.loads(v['data']) for v in reversed(events) if v['data']!='[DONE]');assert final['response_token_ids']==records['s3-'+case+'-c1-r0'][1]['choices'][0]['response_token_ids']
 resource=json.loads(z.read('results/native-resident-controls-http-server-resources/resource-monitor.json'));lifetime=[json.loads(l) for l in z.read('results/native-resident-controls-http-server-resources/resource-vram.jsonl').decode().splitlines()];assert resource['sampled_memory']['samples']==len(lifetime) and resource['sampled_memory']['sampled_peak_adapter_used_bytes']==max(row['used_bytes'] for row in lifetime)
 stop=json.loads(z.read('results/native-resident-controls-http-intentional-stop.json'));assert stop['taskkill_exit']==0 and stop['workflow_exit_code']==0
 assert resource['child_exit_code']==e['terminal']['native-resident-controls-http-server-ledger']['exit_code']
 # Check retained metadata hashes against included sources/data or separately linked archives.
 for name,h in meta['file_sha256'].items():
  rel=name.replace('\\','/').split('5090-study-20261023/',1)[1]
  if rel in z.namelist():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
  elif rel in e['runtime_sha256']:assert e['runtime_sha256'][rel]==h
  elif rel.startswith('results/') and rel.split('/')[-1] in e['linked_archives']:assert e['linked_archives'][rel.split('/')[-1]]==h
  elif rel in old.namelist():assert hashlib.sha256(old.read(rel)).hexdigest()==h,rel
  else:raise AssertionError('Unverified metadata link '+rel)
 refs=[]
 for name in ('vllm-graph-linkfix-evidence.zip','sglang-corrected-ipv4-evidence.zip'):
  pins={'vllm-graph-linkfix-evidence.zip':'8002dc620d88f1234e6a34548fa1f0d64ad71d0c13c8255e74fce9fcd98ca930','sglang-corrected-ipv4-evidence.zip':'c83557d551ccca61d0092566e9e7ad236a356ef4adda1ccbb5549e545acb820b'}
  assert hashlib.sha256((out/name).read_bytes()).hexdigest()==pins[name]
  with zipfile.ZipFile(out/name) as ref:
   assert ref.testzip() is None;ref_manifest=json.loads(ref.read('evidence-manifest.json'))
   for n,h in ref_manifest['files'].items():assert hashlib.sha256(ref.read(n)).hexdigest()==h,n
   candidates=[]
   for n in ref.namelist():
    if n.endswith('-manifest.json'):
     doc=json.loads(ref.read(n))
     if isinstance(doc,dict) and doc.get('status')=='completed' and 'cohorts' in doc:candidates.append((n,doc))
   assert len(candidates)==1;n,doc=candidates[0];pre=n.rsplit('/',1)[0]+'/';refs.append(verify_run(doc,lambda n:ref.read(pre+n)))
 result={'status':'resident_controls_http_evidence_verified','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files_verified':len(e['files']),'strict_gates':gates,'prior_qualified_pairs_exact':20,'measured_raw_sse_requests':16,'measured_ids_exact':16,'stream_gates':2,'worker':worker,'run':run,'cohorts':m['cohorts'],'vs_ordinary_references':compare(run,refs),'two_x_established':False,'scope':'One resident-control B1 short-prompt pilot; raw events/IDs/VRAM verified. No causal gain, repeated same-binary A/B confidence, full-context/broad-quality, true CUDA batching/concurrency, tuned references or2x admission.'}
(out/'native-resident-controls-http-verified.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
