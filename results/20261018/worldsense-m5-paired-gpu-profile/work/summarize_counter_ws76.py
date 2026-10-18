"""Prespecified sparse-pass summaries, with overlaps/units/first calls retained."""
from pathlib import Path
import collections,csv,datetime,hashlib,json
b=Path(__file__).resolve().parent.parent;w=b/'worldsense-paired-gpu-profile';p=w/'capture';load=lambda q:json.loads(q.read_text())
verification=load(p/'verification.json');assert verification['counter_commands']==48 and not verification['speed_eligibility']
lookup={(row['label'],row['request']):row for row in verification['calls']}
def union(intervals):
 total=0;start=None;end=None
 for a,z in sorted(intervals):
  if start is None:start,end=a,z
  elif a>end:total+=end-start;start,end=a,z
  else:end=max(end,z)
 return total+(0 if start is None else end-start)
windows=[];families=[];raw_files={}
for label in ['legacy-1','current-1','current-2','legacy-2']:
 trace=p/(label+'.counters.jsonl');raw_files[label]=hashlib.sha256(trace.read_bytes()).hexdigest();pending=[]
 for line in trace.read_text().splitlines():
  kind,payload=line.split(' ',1);v=json.loads(payload)
  if kind=='metal_counter_profile':pending.append(v);continue
  assert kind=='metal_counter_command';call=lookup[(label,v['request'])];phase='first'if call['warmup']else'followup'
  intervals=[(row['start_raw'],row['end_raw'])for row in pending];duration=sum(z-a for a,z in intervals);covered=union(intervals);span=max(z for a,z in intervals)-min(a for a,z in intervals)
  assert duration>=covered and span>=covered
  base=dict(label=label,phase=phase,model=v['model'],step=v['decode_step'],pid=v['pid'],request=v['request'],context=v['context'],command=v['command'],raw_trace_sha256=raw_files[label])
  windows.append(dict(**base,passes=len(pending),duration_sum_raw=duration,merged_union_raw=covered,overlap_raw=duration-covered,sampled_span_raw=span,gaps_raw=span-covered,original_command_gpu_span_ms=v['gpu_span_ms']))
  by=collections.defaultdict(list)
  for row in pending:
   name=row['op'];normalized=name[:-8]+':raw'if name.endswith(':primary')else name;by[normalized].append(row)
  for name,rows in sorted(by.items()):
   parts=[(row['start_raw'],row['end_raw'])for row in rows];value=sum(z-a for a,z in parts)
   families.append(dict(**base,normalized_family=name,actual_family=rows[0]['op'],passes=len(rows),duration_sum_raw=value,merged_union_raw=union(parts),fraction_of_sampled_interval_duration_sum=value/duration if duration else None))
  pending=[]
 assert not pending
def write(name,rows):
 with(p/name).open('w',newline='')as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
write('command-windows.csv',windows);write('operation-family-windows.csv',families)
comparisons=[]
for legacy,current,pair in [('legacy-1','current-1','first-order'),('legacy-2','current-2','mirrored-order')]:
 for phase in ['first','followup']:
  for model in ['Qwen3.5-4B','Qwen3-1.7B']:
   for step in [4,16,28]:
    control=next(v for v in windows if(v['label'],v['phase'],v['model'],v['step'])==(legacy,phase,model,step));candidate=next(v for v in windows if(v['label'],v['phase'],v['model'],v['step'])==(current,phase,model,step))
    comparisons.append(dict(pair=pair,phase=phase,model=model,step=step,legacy=legacy,current=current,legacy_original_command_gpu_span_ms=control['original_command_gpu_span_ms'],current_original_command_gpu_span_ms=candidate['original_command_gpu_span_ms'],original_command_span_ratio=candidate['original_command_gpu_span_ms']/control['original_command_gpu_span_ms']if control['original_command_gpu_span_ms']else None,legacy_duration_sum_raw=control['duration_sum_raw'],current_duration_sum_raw=candidate['duration_sum_raw'],raw_duration_sum_ratio=candidate['duration_sum_raw']/control['duration_sum_raw']if control['duration_sum_raw']else None,legacy_union_raw=control['merged_union_raw'],current_union_raw=candidate['merged_union_raw']))
write('paired-command-comparisons.csv',comparisons)
rankings=[]
for label in raw_files:
 for phase in ['first','followup']:
  for model in ['Qwen3.5-4B','Qwen3-1.7B']:
   rows=[v for v in families if(v['label'],v['phase'],v['model'])==(label,phase,model)];sums=collections.defaultdict(int);counts=collections.defaultdict(int)
   for row in rows:sums[row['normalized_family']]+=row['duration_sum_raw'];counts[row['normalized_family']]+=row['passes']
   total=sum(sums.values())
   for rank,(name,value)in enumerate(sorted(sums.items(),key=lambda kv:kv[1],reverse=True),1):rankings.append(dict(label=label,phase=phase,model=model,rank=rank,normalized_family=name,passes=counts[name],duration_sum_raw=value,fraction_of_three_selected_steps_interval_duration_sum=value/total if total else None))
write('ranked-families.csv',rankings)
summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=4,Skill_requests=8,model_calls=16,selected_commands=48,compute_intervals=30864,windows_with_overlap=sum(v['overlap_raw']>0 for v in windows),raw_files=raw_files,all_first_followup_windows_and_both_worker_orders_retained=True,no_pooled_acceptance_or_performance_promotion=True,limits='Timestamp durations retain raw units; GPU-span milliseconds are separate. Per-encoder intervals overlap; sums/merged unions/span/gaps remain separate. Shape ranks cover only steps4/16/28 and may omit MPS encoders. Separate medians are not summed. Mirrored order does not establish causality, significance, clocks, bandwidth, ISA, registers, occupancy or physical residency. Reconstructed traced legacy differs from deployed WS10 binary. No throughput eligibility.',top_families=[v for v in rankings if v['rank']<=5])
(p/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print('Prespecified raw-unit/overlap/per-window/family and two-order sparse summaries saved; no speed acceptance')
