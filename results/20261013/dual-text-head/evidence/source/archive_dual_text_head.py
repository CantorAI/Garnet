import pathlib,json,hashlib,tarfile,ast
r=pathlib.Path('/Users/zsdx/garnet-metal-investigation');proof={'checks':{},'profiles':[],'clean_requests':0,'omitted_files':{}}
checks=r/'logs/dual-text-head-checks-20261013';bench=r/'logs/dual-text-head-native-20261013'
for p in (checks,bench):assert json.loads((p/'exit.json').read_text())['exit_code']==0
stages=json.loads((checks/'jobs/statuses.json').read_text());assert len(stages)==16 and all(v['exit_code']==0 and v['interruption'] is None for v in stages)
meta=json.loads((checks/'jobs/metadata.json').read_text())
for name,sha in meta['source_sha256'].items():assert hashlib.file_digest((r/'source'/name).open('rb'),'sha256').hexdigest()==sha
assert hashlib.file_digest((r/'build-resident-kv-inplace/libgarnet.dylib').open('rb'),'sha256').hexdigest()==meta['control_sha256']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
assert hashlib.file_digest((r/'build-dual-text-head/libgarnet.dylib').open('rb'),'sha256').hexdigest()==meta['candidate_sha256']
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 text=(checks/'jobs'/(model+'-private-kv.log')).read_text();assert text.count('private_kv_reference PASS')==8
 heads=[ast.literal_eval(v.split('PASS ',1)[1]) for v in (checks/'jobs'/(model+'-head.log')).read_text().splitlines() if v.startswith('BF16 full vocabulary PASS ')]
 assert len(heads)==2 and all(v['worst_l2_ratio']<=1 and v['worst_max_ratio']<=1 for v in heads)
 proof['checks'][model]={'private_cache_profiles':8,'vocabulary_reference_profiles':heads}
summary=json.loads((bench/'jobs/summary.json').read_text());assert len(summary['profiles'])==6
for profile in summary['profiles']:
 folder=pathlib.Path(profile['evidence']);s=json.loads((folder/'summary.json').read_text());assert len(s['records'])==12
 for value in s['records']:
  needed=([24] if profile['batch']==1 else [22]) if profile['model']=='Qwen3-0.6B' else [26] if profile['model']=='Qwen2.5-Coder-0.5B-Instruct' else [80] if profile['batch']==1 else [32,32,16]
  assert value['prefill_chunk_plan']==needed,(profile['model'],profile['batch'],value['prefill_chunk_plan'])
 for policy in ('control','candidate'):
  for repeat in (0,1):
   worker=folder/(profile['model']+'-b'+str(profile['batch'])+'-'+policy+'-'+str(repeat));assert json.loads((worker/'exit.json').read_text())['exit_code']==0;assert json.loads((worker/'monitor.json').read_text())
 proof['profiles'].append(profile);proof['clean_requests']+=12
proof['libraries']=meta;proof['decision']='Reject: affected text batch-1 decode ratio 0.99867, TTFT fraction 1.00001; no full-model gain. No confirmation or new vLLM study warranted.'
out=r/'dual-text-head-evidence.tar.gz'
with tarfile.open(out,'w:gz') as a:
 for p in (checks,bench,r/'logs/dual-text-head-original-source-20261013'):
  for f in sorted(p.rglob('*')):
   if not f.is_file():continue
   if f.suffix in ('.f32','.rgb','.bin'):
    proof['omitted_files'][str(f.relative_to(r))]={'size':f.stat().st_size,'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()}
   else:a.add(f,arcname='completed/'+str(f.relative_to(r/'logs')))
 m=json.loads((r/'dual-text-head-manifest.json').read_text())
 for name,sha in m['files'].items():
  f=r/'source'/name;assert hashlib.file_digest(f.open('rb'),'sha256').hexdigest()==sha;a.add(f,arcname='candidate-source/'+name)
 for name in ('dual-text-head-source.tar.gz','dual-text-head-manifest.json'):a.add(r/name,arcname='source-history/'+name)
 for name in ('CMakeCache.txt','Testing/Temporary/LastTest.log'):
  a.add(r/'build-dual-text-head'/name,arcname='build-dual-text-head/'+name)
 p=r/'dual-text-head-evidence-proof.json';p.write_text(json.dumps(proof,indent=2));a.add(p,arcname='evidence-proof.json')
 a.add(pathlib.Path(__file__),arcname='source/archive_dual_text_head.py')
print('Archive SHA256',hashlib.file_digest(out.open('rb'),'sha256').hexdigest())
print('Candidate library SHA256',meta['candidate_sha256'])
