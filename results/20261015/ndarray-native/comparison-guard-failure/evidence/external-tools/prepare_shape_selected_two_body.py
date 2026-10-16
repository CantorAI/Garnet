"""Copy the retained serving source, then integrate only selected decode shapes."""
from pathlib import Path
import hashlib,json,shutil,sys
parent=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
assert target.parent==parent and target.name=='shape-selected-two-investigation-20261015' and not target.exists()
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for name,expected in (('two-matrix-probe-20261015','022c70ae014a49126effbaf126ac512bfa576901ce48c390504c6ee44463a373'),
                      ('two-matrix-probe-confirm-20261015','d26e23980ee67d443453039c7a543e9fddbe77421707f3f9f9f8c6a274a61cc6')):
    assert json.loads((parent/'logs'/name/'exit.json').read_text())['exit_code']==0
    assert digest(parent/(name+'.tar.gz'))==expected
    meta=json.loads((parent/'logs'/name/'jobs/metadata.json').read_text())
    for name,sha in meta['source_sha256'].items():assert digest(parent/'source'/name)==sha,name
selection=json.loads((parent/'work/two-matrix-confirmation-comparison.json').read_text())
sys.path.insert(0,str(parent/'source/test2026/metal'))
import make_shape_selected_two_body as factory
assert {(p['batch'],p['output_rows'],p['inner'],p['threads']) for p in selection['selected']}==set(factory.TABLE)
target.mkdir();shutil.copytree(parent/'source',target/'source',symlinks=True);changes={}
for name,operation in (
    ('src/backends/metal/metal_context.mm',factory.context),('src/backends/metal/metal_context.h',factory.header),
    ('src/backends/metal/decoder_session.cpp',factory.session),('src/backends/metal/decoder_session.h',factory.session_header),
    ('src/entry/metal_main.cpp',factory.entry),
    ('test2026/metal/run_matched_garnet.py',factory.worker),('test2026/metal/frame_worker_garnet.py',factory.worker),
    ('test2026/metal/coalesced_bf16_body_test.cpp',factory.primitive)):
    path=target/'source'/name;before=digest(path);path.write_bytes(operation(path.read_text()).encode())
    changes[name]=dict(before=before,after=digest(path))
path=target/'source/src/backends/metal/coalesced_two_body_shader.h'
path.write_bytes(factory.shader((parent/'source/src/backends/metal/coalesced_body_block_shader.h').read_text()).encode())
changes[path.relative_to(target/'source').as_posix()]=dict(before=None,after=digest(path))
(target/'logs').mkdir()
for path in (parent/'logs').iterdir():
    if path.is_dir():(target/'logs'/path.name).symlink_to(path,target_is_directory=True)
for name in ('models','work','comparators'):(target/name).symlink_to(parent/name,target_is_directory=True)
for path in parent.glob('build-*'):
    if path.is_dir():(target/path.name).symlink_to(path,target_is_directory=True)
closure={p.relative_to(target/'source').as_posix():digest(p) for p in (target/'source').rglob('*') if p.is_file()}
result=dict(parent=str(parent),target=str(target),changes=changes,source_sha256=closure,selection=selection,
    scope='Private generated source only; retained source/build untouched; shape selection needs fresh full-model confirmation.',full_goal_met=False)
(target/'preparation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(target=str(target),changed_files=changes,copied_source_files=len(closure),full_goal_met=False)))
