"""Join two audited native trials without deriving a stale vLLM comparison."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
audits=[json.loads((root/name).read_text()) for name in ('verification.json','confirmation-verification.json')]
assert all(a['full_output_requests']==72 and not a['full_goal_met'] for a in audits)
rows=[]
for initial,repeat in zip(*(a['profiles'] for a in audits)):
    model,batch=initial['model'],initial['batch']
    assert (repeat['model'],repeat['batch'])==(model,batch)
    folders=[root/'logs/interleaved-body-native-20261015/native'/(model+'-b'+str(batch)+'-attempt0'),root/'logs/interleaved-body-native-confirm-20261015/jobs'/(model+'-b'+str(batch)+'-attempt0')]
    metas=[json.loads((p/'metadata.json').read_text()) for p in folders]
    for key in ('library_sha256','source_sha256','assets','wide_batch_prefill'):
        assert metas[0][key]==metas[1][key],key
    for policy in ('control','candidate'):
        cases=[json.loads((p/(model+'-b'+str(batch)+'-'+policy+'-case.json')).read_text()) for p in folders]
        assert cases[0]==cases[1]
    rows.append(dict(model=model,batch=batch,decode_ratios=[p['decode_ratio'] for p in (initial,repeat)],ttft_fractions=[p['ttft_fraction'] for p in (initial,repeat)],decode_faster_both=all(p['decode_ratio']>1 for p in (initial,repeat)),ttft_lower_both=all(p['ttft_fraction']<1 for p in (initial,repeat)),rss_increase_bytes=[p['peak_individual_rss_bytes']['candidate']-p['peak_individual_rss_bytes']['control'] for p in (initial,repeat)]))
assert len(rows)==6
result=dict(successful_trials=2,full_output_requests=144,identical_assets_libraries_sources_and_cases=True,profiles=rows,selection='Retain optional text decode candidate; coder control remains preferred. VL decode gain is small and costs about 1 GB; no broad default promotion.',scope='Native-only repeated results. No new vLLM ratio. Existing comparator cohorts remain independently reported.',full_goal_met=False)
(root/'combined-verification.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
