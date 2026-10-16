"""Prospective null-only gate: two fresh unchanged cohorts before candidate collection."""
from pathlib import Path
import json,sys
base=Path(sys.argv[1]).resolve();labels=sys.argv[2:] or ['calibration-initial']
assert labels in (['calibration-initial'],['calibration-initial','calibration-repeat'])
trials=[];identity=None
for label in labels:
    folder=base/label;proof=json.loads((folder/'evidence/manifest.json').read_text());assert len(proof['jobs'])==1
    trial=next(iter(proof['jobs']));assert proof['jobs'][trial]['exit_code']==0
    meta=json.loads((folder/'evidence/logs'/trial/'jobs/metadata.json').read_text())
    assert meta['selected_variants']==[0]
    current={k:meta[k] for k in ('source_sha256','shader_sha256','harness_sha256','variants','hardware','versions','timing_design')}
    if identity is None:identity=current
    else:assert current==identity
    audit=json.loads((folder/'verification.json').read_text())
    assert audit['shape_variant_checks']==24 and audit['failed_admissions']==0 and audit['timed_submissions']==4800
    assert all(c['variant']==0 and c['bits_exact'] and c['passed'] for c in audit['checks'])
    serving=[]
    for batch in (1,2):
        p=next(p for p in audit['profiles'] if (p['check']['batch'],p['check']['tokens'],p['check']['heads'],p['check']['dim'])==(batch,216,12,64))
        serving.append(dict(batch=batch,gpu_fraction=p['gpu_fraction'],wall_fraction=p['wall_fraction'],
            pass_gate=all(.95<=p[k+'_fraction']<=1.05 for k in ('gpu','wall'))))
    trials.append(dict(label=label,serving=serving,pass_gate=all(p['pass_gate'] for p in serving)))
result=dict(trials=trials,repeat_admitted=trials[0]['pass_gate'],
    candidate_probe_admitted=len(trials)==2 and all(t['pass_gate'] for t in trials),
    scope='Unchanged scalar pipeline only, all24 original shapes/numerical gates. Both serving batches require GPU and wall null within5% in two fresh cohorts. No samples filtered, no candidate gain or native inference claim.',full_goal_met=False)
(base/'calibration-decision.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
