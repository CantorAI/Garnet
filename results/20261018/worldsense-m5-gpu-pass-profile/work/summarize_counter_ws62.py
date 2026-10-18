"""Rank selected native pass intervals in raw units; never call them whole-GPU shares."""
from pathlib import Path
import json,csv,collections,datetime
base=Path(__file__).resolve().parent.parent;w=base/'worldsense-gpu-counter-profile';p=w/'capture'
verified=json.loads((p/'verification.json').read_text());assert verified['four_actualWorldSense_Galaxy_native_calls_exact']
requests={int(k):v for k,v in verified['requests'].items()};windows=collections.defaultdict(list);commands={};rows=[]
for line in(p/'candidate-1.counters.jsonl').read_text().splitlines():
 kind,payload=line.split(' ',1);c=json.loads(payload);key=(c['request'],c['decode_step'])
 if kind=='metal_counter_profile':windows[key].append(c)
 else:commands[key]=c
summary=[];groups=collections.defaultdict(lambda:collections.defaultdict(lambda:dict(passes=0,duration_raw=0)))
for key,events in sorted(windows.items()):
 req,step=key;who=requests[req];totals=collections.defaultdict(lambda:dict(passes=0,duration_raw=0));intervals=[]
 for c in events:
  duration=c['end_raw']-c['start_raw'];op=c['op'];totals[op]['passes']+=1;totals[op]['duration_raw']+=duration;intervals.append((c['start_raw'],c['end_raw']))
  rows.append(dict(model=who['model'],request=req,warmup=who['warmup'],repetition=who['repetition'],decode_step=step,context=c['context'],command=c['command'],encoder_index=c['index'],op=op,start_raw=c['start_raw'],end_raw=c['end_raw'],duration_raw=duration,eligible_for_speed=False))
 merged=[]
 for start,end in sorted(intervals):
  if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
  else:merged.append([start,end])
 union=sum(end-start for start,end in merged);total=sum(v['duration_raw']for v in totals.values());span=max(e for _,e in intervals)-min(s for s,_ in intervals)
 ranks=[dict(op=op,**v,fraction_of_sampled_interval_duration_sum=v['duration_raw']/total if total else None)for op,v in sorted(totals.items(),key=lambda item:item[1]['duration_raw'],reverse=True)]
 summary.append(dict(request=req,model=who['model'],warmup=who['warmup'],decode_step=step,custom_passes=len(events),sum_interval_duration_raw=total,union_interval_duration_raw=union,overlap_raw=total-union,span_between_first_last_sample_raw=span,gap_within_sample_span_raw=span-union,original_command_gpu_span_ms=commands[key]['gpu_span_ms'],ranks=ranks))
 for op,v in totals.items():
  g=groups[(who['model'],who['warmup'])][op];g['passes']+=v['passes'];g['duration_raw']+=v['duration_raw']
aggregate=[]
for (model,warmup),ops in sorted(groups.items()):
 total=sum(v['duration_raw']for v in ops.values());aggregate.append(dict(model=model,warmup=warmup,selected_steps=[4,16,28],ranks=[dict(op=op,**v,fraction_of_sampled_interval_duration_sum=v['duration_raw']/total if total else None)for op,v in sorted(ops.items(),key=lambda item:item[1]['duration_raw'],reverse=True)]))
with(p/'encoder-samples.csv').open('w',newline='')as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),windows=summary,first_and_warm_separate_aggregates=aggregate,scope='Raw timestamp units remain raw, no guessed conversion to milliseconds. Ranks are shares of the sum of sampled custom-encoder intervals only; MPS-owned passes may be absent, intervals may overlap, and unobserved gaps are not attributed. First/warm separate; three prespecified steps per reached call, one fixed car Skill, no significance, intrinsic kernel speed, full decode distribution or optimization acceptance claim.')
(p/'counter-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for row in aggregate:print(json.dumps(dict(model=row['model'],warmup=row['warmup'],top_sampled_custom_passes=row['ranks'][:5]),indent=2))
