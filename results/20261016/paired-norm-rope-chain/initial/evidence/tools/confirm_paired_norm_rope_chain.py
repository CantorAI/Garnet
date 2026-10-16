"""Require two complete identical-source chain cohorts; never rescue a failed initial workload."""
from pathlib import Path
import json,sys
initial,repeat=(Path(v).resolve() for v in sys.argv[1:3]);trials=[]
for base in (initial,repeat):
 audit=json.loads((base/'verification.json').read_text());assert audit['complete_learned_profiles']==232 and audit['timing_samples']==9600 and audit['carried_stages']==32
 analysis=json.loads((base/'analysis.json').read_text());proof=json.loads((base/'evidence/manifest.json').read_text());trial=next(iter(proof['jobs']));meta=json.loads((base/'evidence/logs'/trial/'jobs/metadata.json').read_text());trials.append(dict(audit=audit,analysis=analysis,trial=trial,metadata=meta))
assert trials[0]['analysis']['unchanged_confirmation_admitted']
for field in ('source_sha256','generated_sha256','defines','reference_library_sha256','learned_report_sha256','assets','timing_design'):assert trials[0]['metadata'][field]==trials[1]['metadata'][field],field
assert trials[0]['trial']=='paired-norm-rope-chain-20261016' and trials[1]['trial']=='paired-norm-rope-chain-confirm-20261016'
workloads=[]
for a,b in zip(trials[0]['analysis']['workloads'],trials[1]['analysis']['workloads']):
 assert (a['model'],a['kind'])==(b['model'],b['kind']);workloads.append(dict(model=a['model'],kind=a['kind'],initial_pass=a['component_screen_pass'],repeat_pass=b['component_screen_pass'],confirmed=a['component_screen_pass'] and b['component_screen_pass']))
result=dict(workloads=workloads,confirmed_workloads=[dict(model=v['model'],kind=v['kind']) for v in workloads if v['confirmed']],complete_learned_profiles=464,timing_samples=19200,original_stage_output_bits_checked=sum(v['audit']['summary']['numeric']['original_output_bits_checked'] for v in trials),source_assets_and_design_unchanged=True,native_integration_admitted=False,native_performance_proven=False,full_goal_met=False,scope='Two complete unchanged carried-chain component cohorts. Only initially passing and confirmed workloads merit further native investigation; no native model or vLLM speed proof.')
Path(sys.argv[3]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
