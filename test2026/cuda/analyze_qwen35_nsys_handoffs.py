"""Instrumented GPU-idle/API interval intersections; no causal speed estimate."""
import argparse,collections,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('sqlite');p.add_argument('output');a=p.parse_args()
c=sqlite3.connect(a.sqlite);names=dict(c.execute('select id,value from StringIds'))
def merged(intervals):
    result=[]
    for start,end in sorted(intervals):
        if result and start<=result[-1][1]:result[-1][1]=max(result[-1][1],end)
        else:result.append([start,end])
    return result
gpu=merged([row for kind in ('KERNEL','MEMCPY','MEMSET') for row in c.execute(f'select start,end from CUPTI_ACTIVITY_KIND_{kind}')])
first,last=gpu[0][0],gpu[-1][1];idle=[[gpu[i][1],gpu[i+1][0]] for i in range(len(gpu)-1) if gpu[i][1]<gpu[i+1][0]]
def intersection(start,end,intervals):return sum(max(0,min(end,z)-max(start,x)) for x,z in intervals if x<end and z>start)
api=collections.defaultdict(lambda:{'count':0,'clipped_sum_ns':0,'gpu_idle_overlap_ns':0});api_intervals=[]
copy_kind={r[0]:r[2] for r in c.execute('select * from ENUM_CUDA_MEMCPY_OPER')};memory_kind={r[0]:r[2] for r in c.execute('select * from ENUM_CUDA_MEM_KIND')}
copies={r[0]:r[1:] for r in c.execute('select correlationId,copyKind,srcKind,dstKind,bytes,start,end from CUPTI_ACTIVITY_KIND_MEMCPY')}
copy_stats=collections.defaultdict(lambda:{'count':0,'cpu_sum_ns':0,'gpu_sum_ns':0,'gpu_idle_overlap_ns':0,'bytes':0})
for start,end,name,corr in c.execute('select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME'):
    start=max(first,start);end=min(last,end)
    if start>=end:continue
    name=names[name];gap=intersection(start,end,idle);entry=api[name];entry['count']+=1;entry['clipped_sum_ns']+=end-start;entry['gpu_idle_overlap_ns']+=gap;api_intervals.append([start,end])
    if name.startswith('cudaMemcpy') and corr in copies:
        kind,src,dst,nbytes,gs,ge=copies[corr];key=(copy_kind.get(kind,str(kind)),memory_kind.get(src,str(src)),memory_kind.get(dst,str(dst)))
        entry=copy_stats[key];entry['count']+=1;entry['cpu_sum_ns']+=end-start;entry['gpu_sum_ns']+=ge-gs;entry['gpu_idle_overlap_ns']+=gap;entry['bytes']+=nbytes
api_union=merged(api_intervals);covered=sum(intersection(start,end,api_union) for start,end in idle);idle_total=sum(end-start for start,end in idle)
def ms_fields(record):return {k[:-3]+'_ms' if k.endswith('_ns') else k:v/1e6 if k.endswith('_ns') else v for k,v in record.items()}
resources={}
for row in c.execute('select demangledName,registersPerThread,localMemoryPerThread,staticSharedMemory,gridX,gridY,gridZ,blockX,blockY,blockZ from CUPTI_ACTIVITY_KIND_KERNEL'):
    name=names[row[0]]
    if 'Garnet' in name:resources.setdefault(name,set()).add(tuple(row[1:]))
result={'source':str(Path(a.sqlite).resolve()),'gpu_activity_span_ms':(last-first)/1e6,'gpu_idle_between_activities_ms':idle_total/1e6,
    'runtime_api_union_during_gpu_idle_ms':covered/1e6,'gpu_idle_outside_runtime_api_union_ms':(idle_total-covered)/1e6,
    'api_ranked_by_idle_overlap':[{'name':name,**ms_fields(value)} for name,value in sorted(api.items(),key=lambda item:-item[1]['gpu_idle_overlap_ns'])],
    'copies_by_kind':[{'kind':key[0],'source_kind':key[1],'destination_kind':key[2],**ms_fields(value)} for key,value in copy_stats.items()],
    'custom_kernel_resources':{name:[{'registers_per_thread':r[0],'reported_local_bytes_per_thread':r[1],'static_shared_bytes':r[2],'grid':r[3:6],'block':r[6:9]} for r in sorted(values)] for name,values in resources.items()},
    'limitation':'One node-traced request. API intervals can overlap threads and GPU work; intersection with idle is not causation or recoverable-time proof. Profiler overhead, Python/host work and GPU dependencies remain. CUPTI local-memory fields may omit stack frames; use separate binary resource output. No matched timing, bandwidth or speedup projection.'}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('gpu_activity_span_ms','gpu_idle_between_activities_ms','runtime_api_union_during_gpu_idle_ms','gpu_idle_outside_runtime_api_union_ms')},indent=2));print(json.dumps(result['api_ranked_by_idle_overlap'][:10],indent=2));print(json.dumps(result['copies_by_kind'],indent=2))
