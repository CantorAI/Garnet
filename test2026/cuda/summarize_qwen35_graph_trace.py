"""Whole-graph GPU intervals plus uncaptured activity; diagnostic, not a speed forecast."""
import argparse,collections,hashlib,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('sqlite');p.add_argument('output');a=p.parse_args()
source=Path(a.sqlite);output=Path(a.output);assert not output.exists(),'Preserve trace summary'
c=sqlite3.connect(source);names=dict(c.execute('select id,value from StringIds'))
tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
schema={name:[r[1] for r in c.execute(f'pragma table_info("{name}")')] for name in tables if name.startswith('CUPTI_ACTIVITY_KIND_')}
assert 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE' in schema,('Missing whole-graph activities',schema)
assert all(k in schema['CUPTI_ACTIVITY_KIND_GRAPH_TRACE'] for k in ('start','end','deviceId'))
graph_devices=list(c.execute('select distinct deviceId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE'))
assert graph_devices==[(0,)],('Expected only GPU0 graph intervals, no host-boundary graph timestamps',graph_devices)
if 'endDeviceId' in schema['CUPTI_ACTIVITY_KIND_GRAPH_TRACE']:
    assert list(c.execute('select distinct endDeviceId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE'))==[(0,)]

def merge(intervals):
    result=[]
    for start,end in sorted(intervals):
        assert start>0 and end>=start,'Unavailable or invalid GPU activity timestamps'
        if result and start<=result[-1][1]:result[-1][1]=max(result[-1][1],end)
        else:result.append([start,end])
    return result

groups={};all_intervals=[]
for kind in ('GRAPH_TRACE','KERNEL','MEMCPY','MEMSET'):
    table='CUPTI_ACTIVITY_KIND_'+kind
    intervals=list(c.execute(f'select start,end from "{table}"')) if table in tables else []
    all_intervals+=intervals
    groups[kind]={'count':len(intervals),'sum_ms':sum(e-s for s,e in intervals)/1e6,
        'union_ms':sum(e-s for s,e in merge(intervals))/1e6}
gpu=merge(all_intervals);assert gpu
first,last=gpu[0][0],gpu[-1][1]
idle=[[gpu[i][1],gpu[i+1][0]] for i in range(len(gpu)-1) if gpu[i][1]<gpu[i+1][0]]
api=collections.defaultdict(lambda:[0,0,0])
for start,end,name in c.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
    start=max(first,start);end=min(last,end)
    if start>=end:continue
    v=api[names[name]];v[0]+=1;v[1]+=end-start
    v[2]+=sum(max(0,min(end,y)-max(start,x)) for x,y in idle if x<end and y>start)
result={'source':str(source.resolve()),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'schema':schema,'groups':groups,'gpu_activity_span_ms':(last-first)/1e6,
    'gpu_activity_union_ms':sum(e-s for s,e in gpu)/1e6,'between_activity_idle_ms':sum(e-s for s,e in idle)/1e6,
    'api_by_idle_overlap':[{'name':n,'count':v[0],'clipped_sum_ms':v[1]/1e6,'idle_overlap_ms':v[2]/1e6} for n,v in sorted(api.items(),key=lambda i:-i[1][2])],
    'scope':'One instrumented warmed request with whole-graph tracing and CUDA event tracing disabled. Graph activity intervals do not reveal node/kernel costs. Whole graph intervals may include internal gaps. API/idle intersection is not causation or recoverable time; sums differ from unions. Not matched timing or a speed forecast.'}
output.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('groups','gpu_activity_span_ms','gpu_activity_union_ms','between_activity_idle_ms')},indent=2))
