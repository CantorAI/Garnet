"""Compare interleaved warm repeat measurements against the speed target."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);jobs=root/(sys.argv[2] if len(sys.argv)>2 else 'double-monotonic-comparison-20261012')/'jobs';records=[];summary=[]
models=('Qwen3.5-0.8B','Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct')
for model in models:
 for batch in (1,2):
  control=[];optimized=[];all_results=[]
  for suffix in ('control','optimized','control-repeat','repeat'):
   folder=jobs/(model+'-b'+str(batch)+'-'+suffix)
   status=json.loads((folder/'exit.json').read_text(encoding='utf-8'));assert status['exit_code']==0,status
   results=[];process=None
   for line in (folder/'output.log').read_text(encoding='utf-8').splitlines():
    marker,_,payload=line.partition(' ')
    if marker in ('metal_image_result','metal_text_result'):
     v=json.loads(payload);v['model']=model;v['folder']=folder.relative_to(root).as_posix();results.append(v)
    elif marker in ('metal_image_process','metal_text_process'):process=json.loads(payload)
   assert len(results)==2 and process['exit_code']==0,folder
   for v in results:
    durations=v['decode_step_seconds'];assert all(d>0 for d in durations),(folder,'Nonpositive duration')
    assert all(d*1000+0.1>=m for d,m in zip(durations,v['decode_session_ms'])),(folder,'Timer shorter than native steady clock')
    v['process']=process;records.append(v)
   all_results.extend(results)
   sample=next(v for v in results if v['request']==1)
   (control if suffix.startswith('control') else optimized).append(sample)
  expected=all_results[0]
  assert all(v['token_ids']==expected['token_ids'] and v['finished']==expected['finished'] for v in all_results),(model,batch,'Output parity')
  assert all(expected['finished']),(model,batch,'Did not reach EOS')
  def means(values):
   return dict(ttft_seconds=statistics.mean(v['ttft_loaded_seconds'] for v in values),
    total_seconds=statistics.mean(v['total_loaded_seconds'] for v in values),
    tokens_per_second=statistics.mean(v['decode_steps_per_second'] for v in values),
    peak_process_rss_gib=statistics.mean(v['process']['peak_rss_bytes']/2**30 for v in values))
  old=means(control);new=means(optimized)
  speedup=new['tokens_per_second']/old['tokens_per_second'];first_token_fraction=new['ttft_seconds']/old['ttft_seconds']
  summary.append(dict(model=model,batch=batch,control=old,optimized=new,decode_speedup=speedup,ttft_fraction=first_token_fraction,
    decode_target_met=speedup>=2,ttft_target_met=first_token_fraction<=0.5,exact_tokens=True,requests_checked=len(all_results),
    output_tokens=expected['output_tokens'],control_samples=control,optimized_samples=optimized))
output=dict(control_commit='f35f58f',warm_request=1,repeats_per_configuration=2,comparisons=summary,records=records,
    all_batch1_targets_met=all(v['decode_target_met'] and v['ttft_target_met'] for v in summary if v['batch']==1),
    all_batch2_targets_met=all(v['decode_target_met'] and v['ttft_target_met'] for v in summary if v['batch']==2))
(root/'target_comparison.json').write_bytes((json.dumps(output,indent=2)+'\n').encode('utf-8'))
for v in summary:print(json.dumps({k:v[k] for k in ('model','batch','control','optimized','decode_speedup','ttft_fraction','decode_target_met','ttft_target_met')}))
print('All batch-1 targets met:',output['all_batch1_targets_met'])
print('All batch-2 targets met:',output['all_batch2_targets_met'])
