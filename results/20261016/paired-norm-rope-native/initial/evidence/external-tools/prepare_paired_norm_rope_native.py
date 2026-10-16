"""Freeze a new private decode source tree only after two complete unchanged component cohorts."""
from pathlib import Path
import hashlib,json,shutil,sys
import make_paired_norm_rope_native as factory
parent=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();selected=parent/'vector-vision-native-investigation-20261016'
assert target.parent==parent and target.name=='paired-norm-rope-native-investigation-20261016' and not target.exists()
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
decision=json.loads((parent/'validation-audits/paired-norm-rope-chain-confirm-20261016/decision-repeat.json').read_text());assert decision['complete_learned_profiles']==464 and decision['timing_samples']==19200 and decision['original_stage_output_bits_checked']==3028549632 and decision['source_assets_and_design_unchanged']
assert decision['confirmed_workloads'] and all(v['kind']=='decode' for v in decision['confirmed_workloads']);models=sorted(v['model'] for v in decision['confirmed_workloads']);assert len(models)==len(set(models))
assert len(decision['cohorts'])==2
for cohort in decision['cohorts']:
 assert digest(parent/(cohort['name']+'.tar.gz'))==cohort['archive_sha256'] and json.loads((parent/'logs'/cohort['name']/'exit.json').read_text())['exit_code']==0
assert digest(selected/'build-vector-vision-width4/libgarnet.dylib')=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
identities=json.loads((selected/'logs/vector-vision-native-confirm-20261016/jobs/source-identities.json').read_text())
for name,sha in identities.items():assert digest(selected/'source'/name)==sha,name
target.mkdir();shutil.copytree(selected/'source',target/'source',symlinks=True);changes={}
for name,operation in factory.operations(models).items():
 p=target/'source'/name;before=digest(p);original=target/'original-source'/name;original.parent.mkdir(parents=True,exist_ok=True);original.write_bytes(p.read_bytes());p.write_bytes(operation(p.read_text()).encode());changes[name]=dict(before=before,after=digest(p))
added={}
for name in ('make_paired_norm_rope.py','make_paired_norm_rope_native.py','prepare_paired_norm_rope_native.py','paired_norm_rope_primitive.mm','test_paired_norm_rope_dense_reference.py'):
 p=parent/'source/test2026/metal'/name;q=target/'source/test2026/metal'/name;assert not q.exists();q.write_bytes(p.read_bytes());added['test2026/metal/'+name]=digest(q)
(target/'logs').mkdir()
for origin in (parent,selected):
 for p in (origin/'logs').iterdir():
  q=target/'logs'/p.name
  if p.is_dir() and not q.exists():q.symlink_to(p,target_is_directory=True)
for name in ('models','work','comparators'):(target/name).symlink_to(parent/name,target_is_directory=True)
for p in parent.glob('build-*'):
 if p.is_dir():(target/p.name).symlink_to(p,target_is_directory=True)
(target/'build-vector-vision-width4').symlink_to(selected/'build-vector-vision-width4',target_is_directory=True)
closure={p.relative_to(target/'source').as_posix():digest(p) for p in (target/'source').rglob('*') if p.is_file()}
result=dict(parent=str(parent),baseline_source=str(selected/'source'),target=str(target),changes=changes,added_sources=added,source_sha256=closure,baseline_source_sha256=identities,baseline_library_sha256=digest(selected/'build-vector-vision-width4/libgarnet.dylib'),selection=decision,selected_models=models,
 native_admission=dict(unmet_decode_profiles=[dict(model='Qwen3-0.6B',batch=1),dict(model='Qwen2.5-Coder-0.5B-Instruct',batch=1)],minimum_decode_gain=1.01,at_least_one_same_unmet_profile_in_both=True,each_profile_decode_fraction_min=.98,each_profile_ttft_fraction_max=1.02,all_six_profiles_required=True,complete_cohorts=2,all_original_correctness_required=True),
 scope='Private decode-only paired Q/K normalization/RoPE on confirmed model geometry, rows1/2. Original arithmetic and independent writes; profiling/other rows retain original path. Actual counters and full numerical/native checks required. Init/first/warm/RSS separate; no native or matched-vLLM gain established.',full_goal_met=False)
(target/'preparation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(dict(target=str(target),source_files=len(closure),changed_files=len(changes),added_files=len(added),selected_models=models,full_goal_met=False)))
