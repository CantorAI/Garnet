"""Preserve interval sums/unions for one captured request without speed claims."""
import argparse,collections,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('sqlite');p.add_argument('output');a=p.parse_args()
c=sqlite3.connect(a.sqlite);names=dict(c.execute('select id,value from StringIds'))
def union(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([start,end])
    return sum(end-start for start,end in merged)
groups={};all_intervals=[];kernels=[]
for table in ('KERNEL','MEMCPY','MEMSET'):
    intervals=list(c.execute(f'select start,end from CUPTI_ACTIVITY_KIND_{table}'));all_intervals+=intervals
    groups[table]={'count':len(intervals),'sum_ms':sum(e-s for s,e in intervals)/1e6,'union_ms':union(intervals)/1e6}
for start,end,name in c.execute('select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
    kernels.append((names[name],end-start))
stats=collections.defaultdict(lambda:[0,0])
for name,delta in kernels:stats[name][0]+=1;stats[name][1]+=delta
total=sum(delta for _,delta in kernels)
ranked=[{'name':name,'count':value[0],'sum_ms':value[1]/1e6,'fraction_of_summed_kernel_time':value[1]/total}
    for name,value in sorted(stats.items(),key=lambda item:-item[1][1])]
api=collections.defaultdict(lambda:[0,0])
for start,end,name in c.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
    name=names[name];api[name][0]+=1;api[name][1]+=end-start
result={'source':str(Path(a.sqlite).resolve()),'groups':groups,'gpu_activity_union_ms':union(all_intervals)/1e6,
    'gpu_activity_span_ms':(max(e for _,e in all_intervals)-min(s for s,_ in all_intervals))/1e6,
    'ranked_kernels':ranked,'runtime_api':[{'name':n,'count':v[0],'sum_ms':v[1]/1e6} for n,v in sorted(api.items(),key=lambda i:-i[1][1])],
    'limitation':'One instrumented warmed request, node-level graph tracing. Sums/unions/span differ; API waits can include GPU work. ProfilerStart setup is retained separately in API records. Not matched performance, bandwidth, occupancy, residency, causality or speed prediction.'}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['groups','gpu_activity_union_ms','gpu_activity_span_ms']},indent=2))
