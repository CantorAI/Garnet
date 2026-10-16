"""Create a confined source copy; leave all existing sources/builds unchanged."""
from pathlib import Path
import hashlib,json,shutil,sys

parent=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
assert target.parent==parent and target.name=='mlp-command-buffer-investigation-20261015'
assert not target.exists()
prior=parent/'logs/mlp-command-buffer-chained-20261015'
assert json.loads((prior/'exit.json').read_text())['exit_code']==0
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert digest(parent/'mlp-command-buffer-chained-20261015.tar.gz')=='398edcdc8bc6f4b7af2adc53899d6448e93c98b250a33eb7afb02e7d5d19ff5b'
meta=json.loads((prior/'jobs/metadata.json').read_text())
for name,sha in meta['source_sha256'].items():assert digest(parent/'source'/name)==sha,name
sys.path.insert(0,str(parent/'source/test2026/metal'))
from make_mlp_command_buffer_context import context,graph,primitive
target.mkdir();shutil.copytree(parent/'source',target/'source',symlinks=True)
changes={}
for name,operation in (
    ('src/backends/metal/metal_context.mm',context),
    ('src/backends/metal/mlp_prefill_graph_impl.h',graph),
    ('test2026/metal/mlp_prefill_graph_test.cpp',lambda s:primitive(s,chained=True))):
    path=target/'source'/name;before=digest(path)
    path.write_bytes(operation(path.read_text()).encode())
    changes[name]=dict(before=before,after=digest(path))
(target/'logs').mkdir()
for path in (parent/'logs').iterdir():
    if path.is_dir():(target/'logs'/path.name).symlink_to(path,target_is_directory=True)
for name in ('models','work'):
    (target/name).symlink_to(parent/name,target_is_directory=True)
for path in parent.glob('build-*'):
    if path.is_dir():(target/path.name).symlink_to(path,target_is_directory=True)
closure={path.relative_to(target/'source').as_posix():digest(path)
    for path in (target/'source').rglob('*') if path.is_file()}
report=dict(scope='Separate command-buffer source copy; old logs/models/builds linked read-only by workflow. Only the three listed copies are transformed.',
    parent=str(parent),target=str(target),changes=changes,source_sha256=closure,
    baseline_library_sha256=digest(parent/'build-coalesced-bf16-body/libgarnet.dylib'),
    prior_primitive_archive_sha256=digest(parent/'mlp-command-buffer-chained-20261015.tar.gz'),
    full_goal_met=False)
(target/'preparation.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
assert all(digest(parent/'source'/name)==sha for name,sha in meta['source_sha256'].items())
print(json.dumps(dict(target=str(target),copied_source_files=len(closure),changed_files=changes,
    scope='Prepared only; no trained validation or timing result',full_goal_met=False)))
