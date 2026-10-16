"""Confirm only fixed initial all-endpoint component selections using one complete unchanged repeat."""
from pathlib import Path
import hashlib,json,sys
base=Path(sys.argv[1]).resolve();initial=base/'initial';repeat=base/'repeat'

def read(folder,name):return json.loads((folder/name).read_text())
def jobs(folder):return folder/'evidence/logs'/read(folder,'verification.json')['trial']/'jobs'

ia,ra=read(initial,'verification.json'),read(repeat,'verification.json');im=read(jobs(initial),'metadata.json');rm=read(jobs(repeat),'metadata.json')
for key in ('source_sha256','generated_shader_sha256','generated_prepare_sha256','checkpoint_sha256','reference_library_sha256','numeric_design','timing_design','admission'):assert im[key]==rm[key],key
assert (initial/'evidence/config.json').read_bytes()==(repeat/'evidence/config.json').read_bytes()
assert read(jobs(initial),'assets/manifest.json')==read(jobs(repeat),'assets/manifest.json')
assert ia['numeric_checks']==ra['numeric_checks']==1872 and ia['timing_samples']==ra['timing_samples']==6400 and ia['original_word_comparisons']==ra['original_word_comparisons']==553986648
decision_a,decision_b=read(initial,'analysis.json'),read(repeat,'analysis.json');assert decision_a['statistic']==decision_b['statistic']
launch=read(jobs(repeat).parent,'confirmation-launch.json');assert launch['selected_workloads']==[dict(width=64,kind='prefill')] and launch['all_initial_sources_unchanged'] and launch['no_third_favorable_repeat']
assert launch['initial_archive_sha256']==read(initial,'archive-verification.json')['archive_sha256']
for name,key in [('verification.json','initial_audit_sha256'),('analysis.json','initial_analysis_sha256')]:assert hashlib.sha256((initial/name).read_bytes()).hexdigest()==launch[key]
workloads=[]
for a,b in zip(decision_a['workloads'],decision_b['workloads']):
 assert (a['width'],a['kind'])==(b['width'],b['kind']);workloads.append(dict(width=a['width'],kind=a['kind'],initial_pass=a['component_screen_pass'],repeat_pass=b['component_screen_pass'],confirmed=a['component_screen_pass'] and b['component_screen_pass']))
cohorts=[dict(label=label,archive_sha256=read(folder,'archive-verification.json')['archive_sha256'],trial=read(folder,'verification.json')['trial']) for label,folder in [('initial',initial),('repeat',repeat)]]
result=dict(cohorts=cohorts,workloads=workloads,confirmed_workloads=[dict(width=w['width'],kind=w['kind']) for w in workloads if w['confirmed']],complete_numeric_profiles=416,original_word_comparisons=1107973296,timing_samples=12800,source_assets_and_design_unchanged=True,native_validation_required=True,native_performance_proven=False,fresh_vllm_run=False,target_count_update=False,full_goal_met=False,scope='Two complete unchanged isolated primitive cohorts only; preserve all null/comparison endpoints. Initial failures cannot be rescued by repeat. No full-model or matched-vLLM gain.')
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
