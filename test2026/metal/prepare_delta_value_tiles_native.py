"""Freeze a private width64 native tree after complete unchanged primitive confirmation."""
from pathlib import Path
import hashlib,json,shutil,sys
import make_delta_value_tiles_native as factory
from make_delta_value_tiles import shader
parent=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();selected=parent/'vector-vision-native-investigation-20261016'
assert target.parent==parent and target.name=='delta-value-tile64-native-investigation-20261016' and not target.exists()

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

decision=json.loads((parent/'validation-audits/delta-value-tiles-gpu-results/decision-repeat.json').read_text());assert decision['confirmed_workloads']==[dict(width=64,kind='prefill')] and decision['source_assets_and_design_unchanged'] and decision['complete_numeric_profiles']==416 and decision['original_word_comparisons']==1107973296 and decision['timing_samples']==12800
for cohort in decision['cohorts']:
 assert digest(parent/(cohort['trial']+'.tar.gz'))==cohort['archive_sha256']
 assert json.loads((parent/'logs'/cohort['trial']/'jobs/exit.json').read_text())['exit_code']==0
assert digest(selected/'build-vector-vision-width4/libgarnet.dylib')=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
identities=json.loads((selected/'logs/vector-vision-native-confirm-20261016/jobs/source-identities.json').read_text())
for name,sha in identities.items():assert digest(selected/'source'/name)==sha,name
target.mkdir();shutil.copytree(selected/'source',target/'source',symlinks=True);changes={}
for name,operation in factory.OPERATIONS.items():
 p=target/'source'/name;before=digest(p);original=target/'original-source'/name;original.parent.mkdir(parents=True,exist_ok=True);original.write_bytes(p.read_bytes());p.write_bytes(operation(p.read_text()).encode());changes[name]=dict(before=before,after=digest(p))
added={}
for name in ('make_delta_value_tiles.py','make_delta_value_tiles_native.py','prepare_delta_value_tiles_native.py','delta_value_tiles_native.mm'):
 p=parent/'source/test2026/metal'/name;q=target/'source/test2026/metal'/name;assert not q.exists();q.write_bytes(p.read_bytes());added['test2026/metal/'+name]=digest(q)
reference=target/'selected-component/original-recurrence.metal';reference.parent.mkdir();reference.write_bytes(shader((selected/'source/src/backends/metal/metal_context.mm').read_text()).encode())
dependencies={name:digest(parent/'source/test2026/metal'/name) for name in ('verify_vector_vision_native_checks.py','verify_mlp_prefill_native_checks.py')}
assert dependencies=={'verify_vector_vision_native_checks.py':'0e57a8ad01c71e549abce786886cc2050283c85f0fc487e816faef0656d013be','verify_mlp_prefill_native_checks.py':'2eb33f027c120dba530aed406ddb60796bfee9484b2e83c10f8a0cc8ab19d2fa'}
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
result=dict(parent=str(parent),target=str(target),baseline_source=str(selected/'source'),source_sha256=closure,baseline_source_sha256=identities,baseline_library_sha256=digest(selected/'build-vector-vision-width4/libgarnet.dylib'),changes=changes,added_sources=added,original_shader_sha256=digest(reference),audit_dependency_sha256=dependencies,selection=decision,
 native_policy=dict(batch=[1,2],chunk=80,heads=16,key_dim=128,value_dim=128,tile=64,threads_per_group=128,decode_unchanged=True,other_shapes_retain_original=True,original_arithmetic=True,original_buffer_writes=True,extra_weight_storage=False),
 native_admission=dict(warm_vl_ttft_fraction_max=.99,each_batch=True,complete_cohorts=2,each_profile_decode_fraction_min=.98,each_profile_ttft_fraction_max=1.02,all_six_profiles_required=True,all_original_correctness_required=True),
 scope='Private OFF-by-default original-bit width64 recurrent chunk80 only; no retained serving change or complete model speed claim. Fresh32 native checks must complete before any performance cohort; matched-vLLM goal unchanged.',full_goal_met=False)
(target/'preparation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(dict(target=str(target),source_files=len(closure),changed_files=len(changes),added_files=len(added),full_goal_met=False)))
