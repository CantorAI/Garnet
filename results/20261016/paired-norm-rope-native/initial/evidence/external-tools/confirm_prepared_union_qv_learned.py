"""Exactly one unchanged complete QV with original K confirmation; require both original screens, no rescue."""
from pathlib import Path
import hashlib,json,sys
initial,repeat=(Path(v).resolve() for v in sys.argv[1:3]);trials=[]
for base in (initial,repeat):
 audit=json.loads((base/'verification.json').read_text());assert audit['complete_learned_profiles']==116 and audit['timing_samples']==4800
 decision=json.loads((base/'decision.json').read_text());proof=json.loads((base/'evidence/manifest.json').read_text());trial=next(iter(proof['jobs']));meta=json.loads((base/'evidence/logs'/trial/'jobs/metadata.json').read_text());trials.append(dict(audit=audit,decision=decision,trial=trial,metadata=meta))
assert trials[0]['decision']['unchanged_confirmation_admitted']
for field in ('source_sha256','generated_sha256','defines','reference_library_sha256','cpu_report_sha256','timing_design','assets'):assert trials[0]['metadata'][field]==trials[1]['metadata'][field],field
assert trials[0]['trial']=='prepared-union-qv-learned-20261016' and trials[1]['trial']=='prepared-union-qv-learned-confirm-20261016'
launch=json.loads((repeat/'evidence/logs'/trials[1]['trial']/'confirmation-launch.json').read_text());assert launch['all_measured_sources_unchanged']
assert launch['initial_verification_sha256']==hashlib.sha256((initial/'verification.json').read_bytes()).hexdigest() and launch['initial_decision_sha256']==hashlib.sha256((initial/'decision.json').read_bytes()).hexdigest()
models=[]
for a,b in zip(trials[0]['decision']['models'],trials[1]['decision']['models']):
 assert a['model']==b['model'];models.append(dict(model=a['model'],initial_pass=a['component_screen_pass'],repeat_pass=b['component_screen_pass'],confirmed_component=a['component_screen_pass'] and b['component_screen_pass']))
selected=[v['model'] for v in models if v['confirmed_component']];result=dict(models=models,confirmed_models=selected,unchanged_source_context_and_design=True,complete_numerical_profiles=232,retained_timing_samples=9600,native_integration_admitted_for_models=selected,native_performance_proven=False,fresh_vllm_comparison=False,full_goal_met=False,scope='Two complete learned QV with original K component cohorts only. Full graph ownership/liveness, native validation and matched repeated model benchmarks remain required.')
Path(sys.argv[3]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
