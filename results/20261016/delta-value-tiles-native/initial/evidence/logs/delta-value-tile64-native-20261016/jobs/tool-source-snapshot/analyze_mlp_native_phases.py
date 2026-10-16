"""Summarize clocks already measured in the audited native MLP cohort."""
from pathlib import Path
import hashlib,json,statistics,sys

base=Path(sys.argv[1]).resolve();root=base/'evidence'
proof=json.loads((root/'manifest.json').read_text())
audit=json.loads((base/'verification.json').read_text())
assert audit['full_output_requests']==72
native=root/'logs/mlp-prefill-native-20261015/native'
profiles=[]
def read(path):
    info=proof['files'][path.relative_to(base).as_posix()]
    raw=path.read_bytes();assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256']
    return raw.decode()
summary=json.loads(read(native/'summary.json'))
for profile in summary['profiles']:
    folder=native/Path(profile['evidence']).name
    statuses=json.loads(read(folder/'statuses.json'));policies={}
    for policy in ('control','candidate'):
        rows=[]
        for stage in statuses:
            if '-'+policy+'-' not in stage['name']:continue
            prefix='matched_frame_result ' if profile['model']=='Qwen3.5-0.8B' else 'matched_result '
            rows.extend(json.loads(line[len(prefix):]) for line in read(folder/stage['name']/'output.log').splitlines() if line.startswith(prefix))
        warm=[row for row in rows if row['request']>0];assert len(warm)==4
        data=dict(warm_requests=4,ttft_ms=statistics.mean(v['ttft_seconds']*1000 for v in warm),
            prefill_profile={key:statistics.mean(v['prefill_profile'][key] for v in warm) for key in ('prepare_ms','encode_ms','wait_ms','gpu_ms')},
            interpretation='GPU span is a separate overlapping observation; never add it to prepare/encode/wait. Synchronous candidate encode includes graph execution.')
        if profile['model']=='Qwen3.5-0.8B':
            data['stages_ms']={key:statistics.mean(v[key]*1000 for v in warm) for key in ('common_preprocessing_seconds','vision_seconds','assembly_seconds','prefill_wall_seconds','reset_seconds')}
            data['decode_phase_counters_available']=False
        else:
            request_means=[]
            for row in warm:
                count=row['decode_steps_after_first'];phases=row['decode_phases']
                assert len(phases)==len(row['native_decode_ms'])==len(row['decode_step_seconds'])==count and count>0
                request_means.append(dict(
                    **{key:statistics.mean(p[key] for p in phases) for key in ('prepare_ms','encode_ms','wait_ms','gpu_ms')},
                    native_ms=statistics.mean(row['native_decode_ms']),
                    measured_step_ms=1000*statistics.mean(row['decode_step_seconds']),
                    wall_window_per_step_ms=1000*row['decode_window_seconds']/count))
            data['decode_phase_counters_available']=True
            data['decode_per_step_ms_equal_request_means']={key:statistics.mean(v[key] for v in request_means) for key in request_means[0]}
        policies[policy]=data
    profiles.append(dict(model=profile['model'],batch=profile['batch'],policies=policies))
result=dict(source_archive_sha256='9d494b2795485f03def18f463673b28da754fb6b5b99cce8db6c0ac96f7c5242',
    scope='Derived clock descriptions of the rejected synchronous MLP cohort and its retained control. No new runtime execution, causal isolation, latency prediction or vLLM ratio.',profiles=profiles,full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps([dict(model=p['model'],batch=p['batch'],control=p['policies']['control'].get('decode_per_step_ms_equal_request_means',p['policies']['control'].get('stages_ms'))) for p in profiles]))
