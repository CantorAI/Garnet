"""Freeze a new private text-only source tree after complete repeated QKV selection."""
from pathlib import Path
import hashlib,json,shutil,sys
import make_prepared_qkv_native as factory
parent=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();selected=parent/'vector-vision-native-investigation-20261016'
assert target.parent==parent and target.name=='prepared-qkv-native-investigation-20261016' and not target.exists()
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
decision=json.loads((parent/'validation-audits/prepared-union-qkv-learned-confirm-20261016/decision-repeat.json').read_text());assert decision['confirmed_models']==['Qwen3-0.6B'] and decision['complete_numerical_profiles']==232
cohorts=[]
for trial,sha in [('prepared-union-qkv-learned-20261016','890cb0e810115043d3cf066686a9b6b9b100e5892a0c1b90d7c5b88cee69d95a'),('prepared-union-qkv-learned-confirm-20261016','ed28aa1ce94c8c006b51c66839052ad661aba5a4d9fa2a78488fe2421dfa5b32')]:
 assert digest(parent/(trial+'.tar.gz'))==sha and json.loads((parent/'logs'/trial/'exit.json').read_text())['exit_code']==0;cohorts.append(dict(name=trial,sha256=sha))
assert digest(selected/'build-vector-vision-width4/libgarnet.dylib')=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
identities=json.loads((selected/'logs/vector-vision-native-confirm-20261016/jobs/source-identities.json').read_text())
for name,sha in identities.items():assert digest(selected/'source'/name)==sha,name
target.mkdir();shutil.copytree(selected/'source',target/'source',symlinks=True);changes={}
for name,operation in factory.OPERATIONS.items():
 p=target/'source'/name;before=digest(p);original=target/'original-source'/name;original.parent.mkdir(parents=True,exist_ok=True);original.write_bytes(p.read_bytes());p.write_bytes(operation(p.read_text()).encode());changes[name]=dict(before=before,after=digest(p))
added={}
for name in ('make_prepared_union_qkv.py','make_prepared_qkv_native.py','prepare_prepared_qkv_native.py','prepared_union_qkv_primitive.mm','test_prepared_qkv_dense_reference.py'):
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
result=dict(parent=str(parent),baseline_source=str(selected/'source'),target=str(target),changes=changes,added_sources=added,source_sha256=closure,baseline_source_sha256=identities,baseline_library_sha256=digest(selected/'build-vector-vision-width4/libgarnet.dylib'),selection=decision,cohorts=cohorts,
 native_admission=dict(selected_models=['Qwen3-0.6B'],warm_text_ttft_fraction_max=.97,each_batch=True,complete_cohorts=2,decode_fraction_min=.98,all_six_profiles_required=True,all_original_correctness_required=True),
 scope='Private prepared exact QKV for confirmed text geometry only; session-context packing during model setup; all three outputs retained; extra union/scratch storage. Init/first/warm and RSS separate. No native gain established.',full_goal_met=False)
(target/'preparation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(dict(target=str(target),source_files=len(closure),changed_files=len(changes),added_files=len(added),baseline_library_sha256=result['baseline_library_sha256'],full_goal_met=False)))
