"""Correlate a controlled WSL run with a VM-scoped WDDM allocation change.

This does not claim direct Linux-process-to-Windows-PID identity. The VM may
include other persistent resources; preserve the baseline and guest allocator.
"""
import argparse,json,statistics,subprocess
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('before');parser.add_argument('capture');parser.add_argument('phases');parser.add_argument('output')
args=parser.parse_args()
before=[json.loads(x) for x in Path(args.before).read_text().splitlines()]
packets=[json.loads(x) for x in Path(args.capture).read_text().splitlines()]
phases=json.loads(Path(args.phases).read_text())
assert before[-1]['event']==packets[-1]['event']==phases[-1]['event']=='complete'
assert before[0]['adapter']['uuid']==packets[0]['adapter']['uuid']
samples=[p for p in packets if p['event']=='sample']
fields=['DedicatedUsage','SharedUsage','TotalCommitted']
baseline_packet=next(p for p in before if p['event']=='sample')
baseline={r['Name']:r for r in baseline_packet['adapter_rows'] if r['cuda_node_selected']}
def timestamp(name):return next(p['unix_seconds'] for p in phases if p['event']==name)
def window(begin,end):return [p for p in samples if p['query_begin_unix']>=begin and p['query_end_unix']<=end]
steady=window(timestamp('steady_before_begin')+1,timestamp('steady_before_end')-1)
assert len(steady)>=2
maps=[{r['Name']:r for r in p['adapter_rows'] if r['cuda_node_selected']} for p in steady]
candidates={}
for name in set.intersection(*(set(m) for m in maps)):
    if name not in baseline:continue
    value=statistics.median(m[name]['DedicatedUsage'] for m in maps)
    delta=value-int(baseline[name]['DedicatedUsage'])
    if delta>2_000_000_000:candidates[name]=delta
assert len(candidates)==1,('Expected exactly one large VM allocation change',candidates)
counter_name=next(iter(candidates));owner=baseline[counter_name]['process_id']
command=f'Get-Process -Id {owner} | Select-Object Id,ProcessName,Path,StartTime | ConvertTo-Json -Depth 3 -Compress'
process=json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-Command',command],text=True))
assert process['ProcessName'].lower() in ('vmwp','vmmem','vmmemwsl'),process
def metrics(chosen):
    rows=[r for p in chosen for r in p['adapter_rows'] if r['Name']==counter_name]
    return {'samples':len(rows),'bytes':{f:{'min':min(int(r[f]) for r in rows),'median':statistics.median(int(r[f]) for r in rows),'max':max(int(r[f]) for r in rows)} for f in fields} if rows else None}
exit_time=next(p['unix_seconds'] for p in packets if p['event']=='watched_process_exited')
after=window(exit_time+1,packets[-1]['unix_seconds'])
worker=next(p['worker_memory'] for p in phases if p['event']=='steady_before_begin')
assert len(worker)==1
vm_rows=[[r for r in p['adapter_rows'] if r['process_id']==owner] for p in steady]
ambiguous=any(sum(any(int(r[f]) for f in fields) for r in rows)>1 for rows in vm_rows)
result={'model':phases[0]['model'],'adapter':packets[0]['adapter'],'windows_vm_owner':process,
    'direct_linux_process_mapping_verified':False,'guest_pid_namespace':phases[0]['pid_namespace'],
    'guest_worker_steady_allocator':worker[0],
    'guest_worker_final_allocator':next(p['worker_memory'][0] for p in phases if p['event']=='complete'),
    'observed_vm_counter_instance':counter_name,'all_vm_instances_during_steady_hold':vm_rows,
    'multiple_nonzero_vm_instances_ambiguous':ambiguous,'native_equivalent_process_metric_available':False,
    'counter_instance_before_bytes':{f:int(baseline[counter_name][f]) for f in fields},
    'counter_instance_steady_before':metrics(steady),
    'counter_instance_steady_after':metrics(window(timestamp('steady_after_begin')+1,timestamp('steady_after_end')-1)),
    'counter_instance_after_guest_exit':metrics(after),
    'arithmetic_instance_dedicated_change_from_before':candidates[counter_name],
    'scope':'Individual WDDM VM counter instance correlated with a controlled single-vLLM-worker run. Multiple instances are retained, never summed; no direct guest-process memory total is established.',
    'limitations':['VM baseline can include other persistent GPU resources; a full VM identity association has not been independently verified.',
        'Guest allocator allocated/reserved/peak counters exclude some driver/external-library allocations.',
        'Counter instance suffixes can change and multiple nonzero VM instances prevent a reliable native-equivalent process total or baseline-subtracted guest allocation claim.',
        'Formatted WDDM samples can miss transient peaks. These are diagnostic holds, not primary timing data.',
        'Do not rank VM totals or baseline-subtracted estimates as equivalent to native whole-process counters.']}
Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
