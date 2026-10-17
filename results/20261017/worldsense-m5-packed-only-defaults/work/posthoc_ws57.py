from pathlib import Path
import json,datetime,statistics,re
w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-packed-only-defaults');load=lambda p:json.loads(p.read_text())
assert not load(w/'production-speed/analysis.json')['passed']
observations={};system={};first={}
for worker in load(w/'production-speed/cohort-manifest.json'):
 label=worker['label'];data=load(w/'production-speed'/(label+'.json'));observations[label]={};first[label]={}
 for model in ['Qwen3.5-4B','Qwen3-1.7B']:
  calls=[(i,c)for i in data['items']for c in i['result']['model_response'].values()if isinstance(c,dict)and c.get('model_id')==model];warm=[c for i,c in calls if not i['warmup']];assert len(warm)==4
  observations[label][model]={k:statistics.median(c['timings'][k]for c in warm)for k in ['decode_seconds','prefill_seconds','reset_seconds']}
  observations[label][model]['post_decode_residual_seconds']=statistics.median(c['total_seconds']-c['ttft_seconds']-c['timings']['decode_seconds']for c in warm)
  observations[label][model]['decode_profile']={k:statistics.median(c['timings']['decode_profile'][k]for c in warm)for k in ['prepare_ms','encode_ms','gpu_ms','wait_ms']}
  first[label][model]={k:calls[0][1][k]for k in ['ttft_seconds','total_seconds','generated_tokens']}
 size=int(re.search(r'page size of (\d+) bytes',worker['vm_stat_before']).group(1));assert size==int(re.search(r'page size of (\d+) bytes',worker['vm_stat_after']).group(1))
 def counters(text):return {k:int(v)for k,v in re.findall(r'^([^:\n]+):\s+(\d+)\.',text,re.M)}
 before,after=counters(worker['vm_stat_before']),counters(worker['vm_stat_after']);delta={k:after[k]-before[k]for k in ['Swapins','Swapouts','Pageouts','Compressions','Decompressions']};assert all(v>=0 for v in delta.values())
 system[label]=dict(page_size_bytes=size,global_counter_deltas=delta,global_swapin_counter_bytes=delta['Swapins']*size,global_swapout_counter_bytes=delta['Swapouts']*size,**{k:worker[k]for k in ['swap_before','swap_after','power_before','power_after','thermal_before','thermal_after']})
post=dict(descriptive_only=True,models=observations,first_after_initialization=first,system_observations=system,limits='No excluded samples or causal attribution. Per-call residuals before aggregation; component medians nonadditive. GPU spans include gaps and wait overlaps. VM counters system-wide/compressor page equivalents, not process attribution or direct disk I/O; API residency requests are not proof of physical residency.');(w/'posthoc-observations.json').write_text(json.dumps(post,indent=2)+'\n')
print('WS57 descriptive phases/first samples/global VM observations saved; no inference rerun')
