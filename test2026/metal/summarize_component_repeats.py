"""Join two audited private cohorts only after source/config/asset identity checks."""
from pathlib import Path
import json,sys

kind=sys.argv[1];root=Path(sys.argv[2]).resolve();assert kind in ('small-body','async-key')
prefix='small-body-threshold' if kind=='small-body' else 'async-key-graph-prefill'
proof=json.loads((root/'combined/evidence/manifest.json').read_text());audit=json.loads((root/'combined-verification.json').read_text())
names=[prefix+'-20261015',prefix+'-confirm-20261015'];assert set(proof['jobs'])==set(names) and all(v['exit_code']==0 for v in proof['jobs'].values())
assert audit['numeric_checks']==(48 if kind=='small-body' else 96) and audit['timed_executions']==(768 if kind=='small-body' else 1536)
jobs=[root/'combined/evidence/logs'/n/'jobs' for n in names];metas=[json.loads((p/'metadata.json').read_text()) for p in jobs]
keys=['source_sha256','assets','reference_static_library_sha256','hardware','versions']
keys+=['context_generated_sha256','harness_generated_sha256','private_header_sha256','context_macro_defines'] if kind=='small-body' else ['generated_cpp_sha256']
for key in keys:assert metas[0][key]==metas[1][key],key
records=[]
for initial in [p for p in audit['profiles'] if p['trial']==names[0]]:
    model,batch,policy=initial['model'],initial['batch'],initial['policy']
    repeat=next(p for p in audit['profiles'] if (p['trial'],p['model'],p['batch'],p['policy'])==(names[1],model,batch,policy))
    fractions=[p['wall_fraction'] for p in (initial,repeat)]
    records.append(dict(model=model,batch=batch,policy=policy,wall_fractions=fractions,wall_faster_both=all(v<1 for v in fractions)))
result=dict(successful_trials=2,identical_sources_assets_versions_hardware_and_config=True,numeric_checks=audit['numeric_checks'],timed_executions=audit['timed_executions'],profiles=records,scope='Private component repeats only. No full-model performance, vLLM ratio, statistical significance or hardware-cold claim.',full_goal_met=False)
(root/'repeat-summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
