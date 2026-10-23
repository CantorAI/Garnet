"""Partition a completed packed memory session without summing graph peaks."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('session');p.add_argument('output');a=p.parse_args()
events=json.loads(Path(a.session).read_text());assert events[-1]['event']=='complete'
memory=next(e['owned_memory'] for e in events if e['event']=='steady_after_begin')
d=memory['decoder'];s=memory['speculative'];v=s['target_verifier'];vision=memory['vision']
components={'Shared BF16 projection pool':d['projection_weight_pool_bytes'],
    'Shared embedding and tied head':d['embedding_head_shared_bytes'],
    'Vision weights and runtime':vision['trt_allocator_live_bytes']+vision['device_output_bytes'],
    'Canonical decoder state':d['state_bytes'],
    ('Retained target outputs and shared pool' if d.get('shared_output_pool_bytes',0) else 'Retained target graph outputs'):d['engine_output_bytes'],
    'Target TRT runtime':d['trt_runtime_allocator_live_bytes']}
target_known=sum(components.values())-components['Vision weights and runtime']
components['Other target IO buffers']=d['owned_decoder_bytes']-target_known
components['MTP engine and request buffers']=s['owned_speculative_bytes']-v['owned_helper_bytes']
components['Verifier backups and head input']=v['backup_and_head_input_bytes']
if v.get('native_prefix_commit') is not None:
    components['Native prefix commit buffers']=v['native_prefix_commit']['owned_bytes']
    replay_keys=('Native prefix commit buffers',)
else:
    components['Prefix replay graph output']=v['packed_replay_graph']['device_output_bytes']
    components['Prefix replay TRT runtime']=v['packed_replay_graph']['trt_allocator_live_bytes']
    replay_keys=('Prefix replay graph output','Prefix replay TRT runtime')
components['Other verification buffers']=v['owned_helper_bytes']-components['Verifier backups and head input']-sum(components[k] for k in replay_keys)
assert all(n>=0 for n in components.values()) and sum(components.values())==memory['owned_model_bytes']
result={'session':a.session,'phase':'steady_after_begin','owned_model_bytes':memory['owned_model_bytes'],
    'components_bytes':components,'decoder_contexts':[g['contexts'] for g in d['graphs']],
    'scope':'Mutually exclusive retained tensor/runtime allocation accounting; no graph peaks are added. Excludes untracked library, driver and transient allocations; not physical residency or lifetime peak.'}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
