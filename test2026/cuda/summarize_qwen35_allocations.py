"""Summarize diagnostic allocation checkpoints without mixing memory scopes."""
import json,re,sys
from pathlib import Path
source=Path(sys.argv[1]);engines={}
for line in source.read_text(errors='replace').splitlines():
    if '[TRTMemoryCheckpoint]' not in line:continue
    fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
    path=fields.pop('engine');stage=fields.pop('stage')
    record=engines.setdefault(path,{'index':len(engines),'engine':path,'stages':{}})
    record['stages'][stage]={k:int(v) for k,v in fields.items()}
for record in engines.values():
    stages=record['stages'];deltas={}
    for operation in ['engine_deserialize','engine_refit','context_create']:
        before=stages.get('before_'+operation);after=stages.get('after_'+operation)
        if before and after:deltas[operation]=after['device_used_bytes']-before['device_used_bytes']
    record['whole_device_observed_delta_bytes']=deltas
result={'source':str(source),'engine_count':len(engines),
    'scope':'cudaMemGetInfo whole-device snapshots; deltas can include concurrent desktop/driver activity. Context upper bounds are TensorRT estimates, not allocated bytes. Not process peak or exact engine weight ownership.',
    'engines':list(engines.values())}
Path(sys.argv[2]).write_text(json.dumps(result,indent=2))
print('QWEN35_ALLOCATION_SUMMARY_COMPLETE',len(engines))
