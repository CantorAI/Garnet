"""Join controlled session phases to sampled, adapter-resolved process counters."""
import argparse,json,statistics
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('session');parser.add_argument('counters');parser.add_argument('output')
args=parser.parse_args();events=json.loads(Path(args.session).read_text())
packets=[json.loads(line) for line in Path(args.counters).read_text().splitlines()]
assert events[-1]['event']=='complete' and packets[-1]['event']=='complete'
mapping=packets[0]['adapter'];pid=events[0]['process_id']
assert all(e['process_id']==pid for e in events)
samples=[p for p in packets if p['event']=='sample'];ambiguous=[];usable=[]
fields=['DedicatedUsage','SharedUsage','TotalCommitted']
for sample in samples:
    assert sample['target_process_id']==pid
    rows=sample['target_rows']
    assert all(r['process_id']==pid and r['cuda_node_selected'] for r in rows)
    nonzero=[r for r in rows if any(int(r[f]) for f in fields)]
    if len(nonzero)>1:
        ambiguous.append({'query_begin_unix':sample['query_begin_unix'],'rows':rows});continue
    if not rows:continue
    # One nonzero row plus zero duplicate instances needs no addition. Multiple
    # nonzero rows remain ambiguous and are never silently aggregated.
    row=nonzero[0] if nonzero else rows[0]
    usable.append(dict(sample=sample,row=row))

def event_time(name):return next(e['unix_seconds'] for e in events if e['event']==name)
def phase(begin,end,margin=0):
    lo=event_time(begin)+margin;hi=event_time(end)-margin
    chosen=[s for s in usable if s['sample']['query_begin_unix']>=lo and s['sample']['query_end_unix']<=hi]
    return {'samples':len(chosen),'begin_unix':lo,'end_unix':hi,
        'counter_names':sorted({s['row']['Name'] for s in chosen}),
        'bytes':{f:{'min':min(int(s['row'][f]) for s in chosen),'median':statistics.median(int(s['row'][f]) for s in chosen),
            'max':max(int(s['row'][f]) for s in chosen)} for f in fields} if chosen else None}
phases={'load_and_build':phase('load_begin','loaded'),
    'steady_before':phase('steady_before_begin','steady_before_end',1),
    'inference':phase('inference_begin','inference_end'),
    'steady_after':phase('steady_after_begin','steady_after_end',1),
    'after_model_close':phase('closed_begin','closed_end',1)}
if events[0].get('loading_observed') is False:
    phases['load_and_build']={'samples':0,'bytes':None,'limitation':'Model loaded before controlled session; loading/build memory not observed.'}
owned=[e['owned_memory']['owned_model_bytes'] for e in events if 'owned_memory' in e and e['event']!='loaded']
result={'engine':events[0].get('engine','garnet'),'model':events[0]['model'],'profile':next(e['profile'] for e in events if e['event']=='loaded'),
    'native_build_policy':events[0].get('native_build_policy',{}),
    'process_id':pid,'adapter':mapping,'phases':phases,
    'largest_observed_bytes':{f:max(int(s['row'][f]) for s in usable) for f in fields} if usable else None,
    'owned_bytes_after_warmup':sorted(set(owned)),'ambiguous_nonzero_duplicate_samples':ambiguous,
    'target_counter_missing_samples':sum(not s['target_counter_present'] for s in samples),
    'scope':'WDDM per-process formatted counters on PCI/CUDA-LUID-matched GPU0 node; target process only. Largest observation is not a true lifetime peak.',
    'limitations':['Formatted-provider update cadence and query latency can miss transient peaks.',
        'Steady/closed phases exclude one second at both boundaries; short inference phases can have few or no enclosed samples.',
        'Dedicated, shared and committed counters are distinct; they are not substituted for allocator-owned bytes.',
        'Multiple nonzero duplicate instances are ambiguous and excluded, not summed.',
        'Polling/phase holds invalidate latency comparisons; primary cohorts are separate.']}
Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
