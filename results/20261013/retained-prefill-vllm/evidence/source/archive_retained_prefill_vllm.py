import hashlib,json,tarfile
from pathlib import Path
r=Path('/Users/zsdx/garnet-metal-investigation');names=['retained-prefill-vllm-20261013']
for n in names:assert json.loads((r/'logs'/n/'exit.json').read_text())['exit_code']==0,n
p=r/'logs/retained-prefill-vllm-20261013/jobs';summary=json.loads((p/'summary.json').read_text());assert len(summary['profiles'])==6 and len(summary['dtype_audits'])==3
proof={'profiles':[],'omitted_files':{},'counts_as_default_bf16_speedup':False}
for audit in summary['dtype_audits']:
 worker=audit['report']['workers'][0];assert set(worker['actual_parameter_tensors'])=={'mlx.core.float32'}
 assert worker['runner_kv_dtype']=='mlx.core.float32' and set(worker['activation_dtype_probes'].values())=={'mlx.core.float32'}
for profile in summary['profiles']:
 folder=Path(profile['evidence']);data=json.loads((folder/'summary.json').read_text());assert len(data['records'])==12 and data['exact_complete_outputs']
 assert not data['counts_as_default_bf16_speedup'];metadata=json.loads((folder/'metadata.json').read_text());assert metadata['native_library_sha256']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
 for value in data['records']:
  if value['backend']=='garnet':
   model=profile['model'];batch=profile['batch']
   expected=([24] if batch==1 else [22]) if model=='Qwen3-0.6B' else [26] if model=='Qwen2.5-Coder-0.5B-Instruct' else [80] if batch==1 else [32,32,16]
   assert value['prefill_chunk_plan']==expected
 for runtime in ('garnet','vllm-metal'):
  for repeat in (0,1):
   worker=folder/(runtime+'-'+str(repeat));assert json.loads((worker/'exit.json').read_text())['exit_code']==0
   monitor=json.loads((worker/'monitor.json').read_text());assert monitor and all(v['guard_passed'] for v in monitor)
   assert (folder/(runtime+'-'+str(repeat)+'-after-processes.txt')).exists()
 if profile['model']=='Qwen3.5-0.8B':
  ids=[]
  for value in data['records']:
   if value['backend']=='garnet':assert value['vision_reencoded']
   else:ids+=value['encoder_cache_identities']
  assert len(ids)==len(set(ids))
 proof['profiles'].append({k:v for k,v in data.items() if k!='records'})
proof['clean_requests']=72;proof['dtype_audits']=summary['dtype_audits']
out=r/'retained-prefill-vllm-evidence.tar.gz'
with tarfile.open(out,'w:gz') as a:
 for name in names:
  for f in sorted((r/'logs'/name).rglob('*')):
   if not f.is_file():continue
   if f.suffix in ('.f32','.rgb','.bin'):
    proof['omitted_files'][str(f.relative_to(r))]={'size':f.stat().st_size,'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()}
   else:a.add(f,arcname='completed/'+str(f.relative_to(r/'logs')))
 for f in (r/'retained-prefill-vllm-source.tar.gz',r/'retained-prefill-vllm-manifest.json'):a.add(f,arcname='source-history/'+f.name)
 for name in ('compare_retained_prefill_vllm.py','run_retained_prefill_vllm_profiles.py','audit_comparator_dtypes.py','run_matched_garnet.py','run_matched_vllm.py','frame_worker_garnet.py','frame_worker_vllm.py','run_matched_frame_process.py','benchmark_matched_process.py'):
  a.add(r/'source/test2026/metal'/name,arcname='source/'+name)
 a.add(r/'resident-kv-vllm-hardware.json',arcname='hardware.json')
 a.add(r/'source/xModel/qwen/metal_dense/decoder.py',arcname='source/metal_dense_decoder.py')
 a.add(Path(__file__),arcname='source/archive_retained_prefill_vllm.py')
 f=r/'retained-prefill-vllm-evidence-proof.json';f.write_text(json.dumps(proof,indent=2));a.add(f,arcname='evidence-proof.json')
print('Archive SHA256',hashlib.file_digest(out.open('rb'),'sha256').hexdigest())
