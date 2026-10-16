"""Apply the predeclared first-cohort screen; preserve complete negative timings and memory."""
from pathlib import Path
import json,sys
base=Path(sys.argv[1]).resolve();audit=json.loads((base/'comparison/verification.json').read_text())
assert audit['full_output_requests']==72 and audit['actual_ndarray_prefill_routes']
assert audit['archive_sha256']=='7ed9d48e71a3723cf84e62fc95170bbee52cf614d90ef8fd7c93df338dc92c33'
profiles=[]
for r in audit['profiles']:
    m=r['means'];first=r['first_request_after_initialization'];rss=r['peak_individual_rss_bytes']
    nd=186 if (r['model'],r['batch'])==('Qwen3.5-0.8B',1) else 0
    profiles.append(dict(**r,ndarray_prefill_calls=nd,ndarray_decode_calls=0,
        aggregate_decode_tps={k:v['decode_tps']*r['batch'] for k,v in m.items()},
        warm_ttft_reduction_fraction=1-r['ttft_fraction'],decode_change_fraction=r['decode_ratio']-1,
        added_peak_individual_rss_gib=(rss['candidate']-rss['control'])/2**30,
        first_ttft_increase_ms=first['candidate']['ttft_ms']-first['control']['ttft_ms']))
vl=next(p for p in profiles if (p['model'],p['batch'])==('Qwen3.5-0.8B',1))
assert vl['ttft_fraction']>.97
result=dict(profiles=profiles,request_confirmation=False,promote=False,repeat_sought=False,
    reason='Initial complete cohort fails the predeclared >=3% VL B1 warm-TTFT screen.',
    retained_library_sha256='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1',
    rejected_library_sha256=audit['checks']['candidate_library_sha256'],fresh_vllm_run=False,target_count_update=False,
    scope='Native complete-model cohort only. Negative first/warm/decode/RSS values retained. No new vLLM ratio or joint target pass; no favorable-repeat search.',full_goal_met=False)
(base/'decision-analysis.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
