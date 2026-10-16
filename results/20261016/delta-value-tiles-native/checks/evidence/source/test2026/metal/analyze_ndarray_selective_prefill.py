"""Apply the predeclared selective screen to independently audited complete cohorts."""
from pathlib import Path
import json,sys
base=Path(sys.argv[1]).resolve();labels=sys.argv[2:] or ['initial']
assert labels in (['initial'],['initial','repeat'])
trials=[];identities=None
for label in labels:
    frozen=json.loads((base/label/'archive-verification.json').read_text())
    expected={'initial':'f70a9eb265f99f768c0b23cbf033cadbffb758211febaaae1f5bc887638bf05f',
              'repeat':'a4de900f6d79249ed4ef69c5ce6c5d51fc09fdcf50f7199ea1f570f4573a62b4'}
    assert frozen['archive_sha256']==expected[label]
    name='ndarray-selective-prefill-'+('confirm-' if label=='repeat' else '')+'20261016'
    metadata=json.loads((base/label/'evidence/logs'/name/'jobs/metadata.json').read_text())
    current={k:metadata[k] for k in ('source_sha256','assets','context_generated_sha256','harness_generated_sha256','private_header_sha256','context_macro_defines')}
    if identities is None:identities=current
    else:assert current==identities,'Cohorts must use identical sources/assets/macros'
    audit=json.loads((base/label/'verification.json').read_text());assert audit['numeric_checks']==72 and audit['verified_files']==62
    assert all(c['passed'] for p in audit['profiles'] if p['policy']==2 for c in p['checks'])
    assert not any(p['policy']==2 for p in audit['rejected_profiles'])
    groups=[]
    for batch in (1,2):
        profiles={p['policy']:p for p in audit['profiles'] if p['model']=='Qwen3.5-0.8B' and p['batch']==batch}
        assert set(profiles)=={1,2,3}
        candidate,null=profiles[2],profiles[3]
        assert all(c['actual_ndarray_dispatches']==1 for c in candidate['checks'])
        assert all(c['actual_ndarray_dispatches']==0 and c['bitwise_equal'] for c in null['checks'])
        gain=all(candidate[k+'_fraction']<=.95 for k in ('gpu','wall'))
        calibrated=all(.95<=null[k+'_fraction']<=1.05 for k in ('gpu','wall'))
        groups.append(dict(batch=batch,gpu_fraction=candidate['gpu_fraction'],wall_fraction=candidate['wall_fraction'],
            null_gpu_fraction=null['gpu_fraction'],null_wall_fraction=null['wall_fraction'],
            screen_pass=gain and calibrated))
    trials.append(dict(label=label,profiles=groups,screen_pass=all(p['screen_pass'] for p in groups)))
result=dict(trials=trials,initial_screen_pass=trials[0]['screen_pass'],repeat_admitted=trials[0]['screen_pass'],
    native_integration_admitted=len(trials)==2 and all(t['screen_pass'] for t in trials),
    promote=False,target_count_update=False,full_goal_met=False,
    scope='Original trained component screen only. Native admission is permission to investigate, not proof of model/vLLM gain.')
(base/('decision-'+labels[-1]+'.json')).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result))
