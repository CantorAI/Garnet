"""Freeze a new private source tree after two audited vector cohorts select width4."""
from pathlib import Path
import hashlib,json,shutil,sys
import make_vector_vision_native as factory
parent=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
assert target.parent==parent and target.name=='vector-vision-native-investigation-20261016' and not target.exists()
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
selection=json.loads((parent/'validation-audits/vector-vision-calibrated/decision-repeat.json').read_text())
assert selection['native_integration_vector_width']==4 and len(selection['trials'])==2 and not selection['promote']
cohorts=[]
for name,sha in [('vector-vision-calibrated-20261016','bcdc7fd237ff515d56604ef89200cecd5cffaff44358d60bf26d0e282f6e5fec'),('vector-vision-calibrated-confirm-20261016','e0caf4a4e40e5d2b8b72984734f2509b5671ef87f92f9f6276e4f00ceefbbf77')]:
    assert digest(parent/(name+'.tar.gz'))==sha
    assert json.loads((parent/'logs'/name/'exit.json').read_text())['exit_code']==0
    cohorts.append(dict(name=name,archive_sha256=sha))
assert digest(parent/'build-coalesced-bf16-body/libgarnet.dylib')=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
selected=parent/'logs/vector-vision-calibrated-confirm-20261016/jobs/vector-vision.metal'
assert selected.read_bytes()==(parent/'logs/vector-vision-calibrated-20261016/jobs/vector-vision.metal').read_bytes()
target.mkdir();shutil.copytree(parent/'source',target/'source',symlinks=True);changes={}
for name,operation in factory.OPERATIONS.items():
    path=target/'source'/name;before=digest(path);path.write_bytes(operation(path.read_text()).encode());changes[name]=dict(before=before,after=digest(path))
name='src/backends/metal/serial_vision_prefetch_shader.h';path=target/'source'/name;before=digest(path)
path.write_bytes(factory.shader(path.read_text(),selected.read_text()).encode());changes[name]=dict(before=before,after=digest(path))
(target/'logs').mkdir()
for path in (parent/'logs').iterdir():
    if path.is_dir():(target/'logs'/path.name).symlink_to(path,target_is_directory=True)
for name in ('models','work','comparators'):(target/name).symlink_to(parent/name,target_is_directory=True)
for path in parent.glob('build-*'):
    if path.is_dir():(target/path.name).symlink_to(path,target_is_directory=True)
closure={p.relative_to(target/'source').as_posix():digest(p) for p in (target/'source').rglob('*') if p.is_file()}
result=dict(parent=str(parent),target=str(target),changes=changes,source_sha256=closure,selection=selection,cohorts=cohorts,
 selected_shader_sha256=digest(selected),baseline_library_sha256=digest(parent/'build-coalesced-bf16-body/libgarnet.dylib'),
 native_admission=dict(warm_vl_ttft_fraction_max=.98,each_batch=True,complete_cohorts=2,decode_fraction_min=.98,all_original_correctness_required=True),
 scope='Private ordered float4 vision context only; FP32 arithmetic and scalar fallback. No new weights, preprocessing, prefix reuse, or decode algorithm. No inference gain established.',full_goal_met=False)
(target/'preparation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(target=str(target),copied_source_files=len(closure),changes=changes,full_goal_met=False)))
