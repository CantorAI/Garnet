"""Extract TensorRT-reported context bounds from a separate memory diagnostic."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('native_log',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
assert not a.output.exists(),'Preserve context requirement evidence'
records=[]
pattern=re.compile(r'\[TRTMemoryCheckpoint\] stage=(\S+) device=(\d+) device_used_bytes=(\d+) device_total_bytes=(\d+) context_device_memory_upper_bound_bytes=(\d+) context_count=(\d+) engine=(.*)')
for line in a.native_log.read_text(encoding='utf-8',errors='replace').splitlines():
    match=pattern.search(line)
    if match:
        stage,device,used,total,bound,count,engine=match.groups()
        records.append(dict(stage=stage,device=int(device),device_used_bytes=int(used),device_total_bytes=int(total),
            context_upper_bound_bytes=int(bound),context_count=int(count),engine=engine))
created={}
for record in records:
    if record['stage']=='after_context_create':created[record['engine']]=record
assert created and all(r['engine'] and r['context_count']>0 for r in created.values()),'Missing identifiable context creation evidence'
result={'source':a.native_log.name,'source_sha256':hashlib.sha256(a.native_log.read_bytes()).hexdigest(),
    'all_checkpoints':records,'per_engine_last_context_create':list(created.values()),
    'sum_context_upper_bounds_bytes_not_measured_allocation':sum(r['context_upper_bound_bytes']*r['context_count'] for r in created.values()),
    'largest_single_context_upper_bound_bytes':max(r['context_upper_bound_bytes'] for r in created.values()),
    'scope':'TensorRT getDeviceMemorySizeV2 upper bounds at context creation during isolated memory diagnostics. Bounds are not allocated-byte measurements or predicted savings. Device-used snapshots cover the whole device. No latency or workspace-sharing safety claim.'}
a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({key:value for key,value in result.items() if key not in ('all_checkpoints','per_engine_last_context_create')},indent=2))
