"""Derive separate warm, setup and first-use observations from archived rows."""
import json,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();result=[]
for folder in sorted((r/'evidence/logs').glob('webp-thread-inference*')):
    top=json.loads((folder/'jobs/summary.json').read_text())
    for profile in top['profiles']:
        job=folder/'jobs'/Path(profile['evidence']).name
        data=json.loads((job/'summary.json').read_text());rows=data['records']
        for backend in ('garnet','vllm-metal'):
            observations={}
            for policy in ('selected','threaded'):
                values=[v for v in rows if v['backend']==backend and v['frontend_policy']==policy]
                warm=[v for v in values if v['request']>0];first=[v for v in values if v['request']==0]
                assert len(values)==6 and len(warm)==4 and len(first)==2
                mean=lambda selected,key:statistics.mean(v[key] for v in selected)
                observations[policy]=dict(warm_ttft_ms=mean(warm,'ttft_seconds')*1000,
                    warm_tps_per_row=mean(warm,'decode_wall_steps_per_second'),
                    warm_tps_aggregate=mean(warm,'decode_wall_steps_per_second')*profile['batch'],
                    warm_preparation_ms=mean(warm,'common_preprocessing_seconds')*1000,
                    first_request_ttft_ms=mean(first,'ttft_seconds')*1000,
                    worker_initialization_seconds=mean(first,'initialization_seconds'),
                    frontend_setup_seconds=mean(first,'frontend_setup_seconds'),
                    peak_child_rss_bytes=statistics.mean(v['process']['peak_rss_bytes'] for v in first))
            baseline=observations['selected'];candidate=observations['threaded']
            result.append(dict(run=folder.name,batch=profile['batch'],backend=backend,observations=observations,
                warm_ttft_reduction_percent=(1-candidate['warm_ttft_ms']/baseline['warm_ttft_ms'])*100,
                warm_decode_ratio=candidate['warm_tps_per_row']/baseline['warm_tps_per_row'],
                first_request_ttft_fraction=candidate['first_request_ttft_ms']/baseline['first_request_ttft_ms']))
(r/'observations.json').write_bytes((json.dumps(dict(scope='Separate measured intervals; no synthetic hardware-cold sum',profiles=result),indent=2)+'\n').encode())
print(json.dumps(dict(observations=len(result))))
