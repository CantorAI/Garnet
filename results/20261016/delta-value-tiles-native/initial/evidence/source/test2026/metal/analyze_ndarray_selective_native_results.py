"""Apply both-VL native screen without changing the full repeated matched-vLLM goal."""
from pathlib import Path
import json,sys
base=Path(sys.argv[1]).resolve();labels=sys.argv[2:] or ['initial']
assert labels in (['initial'],['initial','repeat'])
trials=[];candidate=None;sources=None
for label in labels:
    folder=base/label;audit=json.loads((folder/'verification.json').read_text())
    assert audit['full_output_requests']==72 and audit['actual_ndarray_prefill_routes']
    assert audit['regenerated_native_comparators']==6 and audit['retained_common_half_counters_verified']==72
    archive=json.loads((folder/'archive-verification.json').read_text())
    assert archive['archive_sha256']==audit['archive_sha256']
    identity=audit['checks']['candidate_library_sha256']
    if candidate is None:candidate=identity;sources=audit['checks']['source_manifest']
    else:assert candidate==identity and sources==audit['checks']['source_manifest']
    profiles=[]
    for r in audit['profiles']:
        m=r['means'];rss=r['peak_individual_rss_bytes'];first=r['first_request_after_initialization']
        profiles.append(dict(**r,aggregate_decode_tps={k:v['decode_tps']*r['batch'] for k,v in m.items()},
            warm_ttft_reduction_fraction=1-r['ttft_fraction'],decode_change_fraction=r['decode_ratio']-1,
            added_mean_worker_peak_rss_gib=(rss['candidate']-rss['control'])/2**30,
            first_ttft_increase_ms=first['candidate']['ttft_ms']-first['control']['ttft_ms']))
    assert len(profiles)==6
    active=[p for p in profiles if p['model']=='Qwen3.5-0.8B'];assert {p['batch'] for p in active}=={1,2}
    trials.append(dict(label=label,archive_sha256=audit['archive_sha256'],profiles=profiles,
        active_warm_ttft_screen_pass=all(p['ttft_fraction']<=.97 for p in active)))
regressions=[]
if len(trials)==2:
    for first in trials[0]['profiles']:
        second=next(p for p in trials[1]['profiles'] if (p['model'],p['batch'])==(first['model'],first['batch']))
        if first['decode_ratio']<=.98 and second['decode_ratio']<=.98:
            regressions.append(dict(model=first['model'],batch=first['batch']))
result=dict(trials=trials,initial_screen_pass=trials[0]['active_warm_ttft_screen_pass'],
    repeat_admitted=trials[0]['active_warm_ttft_screen_pass'],
    native_confirmation_pass=len(trials)==2 and all(t['active_warm_ttft_screen_pass'] for t in trials) and not regressions,
    repeated_decode_regressions=regressions,candidate_library_sha256=candidate,
    retained_library_sha256='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1',
    promote=False,fresh_vllm_run=False,target_count_update=False,full_goal_met=False,
    scope='Complete native-only cohorts. >=3% TTFT reduction in BOTH active VL profiles; unchanged fresh confirmation and no repeated >=2% decode regression. First/init/warm/mean worker peak RSS separate. No new vLLM ratio or goal pass.')
(base/('decision-'+labels[-1]+'.json')).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='trials'}))
