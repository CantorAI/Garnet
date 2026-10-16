"""Apply prospective two-cohort native retention gate; preserve full goal as unmet."""
from pathlib import Path
import json,sys
base=Path(sys.argv[1]).resolve();labels=sys.argv[2:] or ['initial'];assert labels in (['initial'],['initial','repeat'])
trials=[];candidate=None;sources=None
for label in labels:
    folder=base/label;audit=json.loads((folder/'verification.json').read_text())
    assert audit['full_output_requests']==72 and audit['actual_vector_vision_routes'] and audit['regenerated_native_comparators']==6
    assert json.loads((folder/'archive-verification.json').read_text())['archive_sha256']==audit['archive_sha256']
    identity=audit['checks']['candidate_library_sha256']
    if candidate is None:candidate=identity;sources=audit['checks']['source_manifest']
    else:assert candidate==identity and sources==audit['checks']['source_manifest']
    profiles=[]
    for row in audit['profiles']:
        m=row['means'];rss=row['peak_individual_rss_bytes'];first=row['first_request_after_initialization']
        profiles.append(dict(**row,aggregate_decode_tps={k:v['decode_tps']*row['batch'] for k,v in m.items()},
          warm_ttft_reduction_fraction=1-row['ttft_fraction'],decode_change_fraction=row['decode_ratio']-1,
          added_mean_worker_peak_rss_gib=(rss['candidate']-rss['control'])/2**30,
          first_ttft_increase_ms=first['candidate']['ttft_ms']-first['control']['ttft_ms']))
    assert len(profiles)==6
    active=[p for p in profiles if p['model']=='Qwen3.5-0.8B'];assert {p['batch'] for p in active}=={1,2}
    trials.append(dict(label=label,archive_sha256=audit['archive_sha256'],profiles=profiles,
      active_warm_ttft_screen_pass=all(p['ttft_fraction']<=.98 for p in active),
      decode_screen_pass=all(p['decode_ratio']>=.98 for p in profiles)))
passed=lambda t:t['active_warm_ttft_screen_pass'] and t['decode_screen_pass']
result=dict(trials=trials,repeat_admitted=passed(trials[0]),
 native_confirmation_pass=len(trials)==2 and all(passed(t) for t in trials),candidate_library_sha256=candidate,
 retained_library_sha256='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1',
 promote=False,fresh_vllm_run=False,target_count_update=False,full_goal_met=False,
 scope='Native-only full cohorts: >=2% warm TTFT reduction BOTH VL batches, decode fraction>=.98 every profile/cohort, original correctness, unchanged confirmation. Init/first/warm/RSS separate; no fresh vLLM ratio or goal pass.')
(base/('decision-'+labels[-1]+'.json')).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='trials'}))
