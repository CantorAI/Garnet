"""Confined native shape80 candidate after two frozen audited component cohorts."""
from pathlib import Path
import hashlib,json,shutil,sys
parent=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
assert target.parent==parent and target.name=='ndarray-native-investigation-20261015' and not target.exists()
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for trial,sha in [('ndarray-prefill-repaired-20261015','2675b4e32ddc1b3f0fab8c4930237ca3a25a8a45f14c4d8dfeebbaf2b33bd263'),
                  ('ndarray-prefill-confirm-20261015','d658c582f33541f953e225599af130275b7cb23af40c748ec8fbf25543daec35')]:
    assert json.loads((parent/'logs'/trial/'exit.json').read_text())['exit_code']==0
    assert digest(parent/(trial+'.tar.gz'))==sha
    meta=json.loads((parent/'logs'/trial/'jobs/metadata.json').read_text())
    for name,identity in meta['source_sha256'].items():assert digest(parent/'source'/name)==identity,name
selection=json.loads((parent/'work/ndarray-prefill-decision.json').read_text())
assert selection['confirmation_complete'] and selection['qualified']==[dict(model='Qwen3.5-0.8B',batch=1,policy=2)]
sys.path.insert(0,str(parent/'source/test2026/metal'));import make_ndarray_native as factory
target.mkdir();shutil.copytree(parent/'source',target/'source',symlinks=True);changes={}
for name,operation in [('src/backends/metal/metal_context.mm',factory.context),('src/backends/metal/metal_context.h',factory.header),
    ('src/backends/metal/decoder_session.cpp',factory.session),('src/backends/metal/decoder_session.h',factory.session_header),
    ('src/entry/metal_main.cpp',factory.entry),('xModel/qwen35/vl_0_8b/metal_decoder.py',factory.decoder),
    ('test2026/metal/run_matched_garnet.py',factory.worker),('test2026/metal/frame_worker_garnet.py',factory.worker)]:
    path=target/'source'/name;before=digest(path);path.write_bytes(operation(path.read_text()).encode());changes[name]=dict(before=before,after=digest(path))
(target/'logs').mkdir()
for path in (parent/'logs').iterdir():
    if path.is_dir():(target/'logs'/path.name).symlink_to(path,target_is_directory=True)
for name in ('models','work','comparators'):(target/name).symlink_to(parent/name,target_is_directory=True)
for path in parent.glob('build-*'):
    if path.is_dir():(target/path.name).symlink_to(path,target_is_directory=True)
closure={p.relative_to(target/'source').as_posix():digest(p) for p in (target/'source').rglob('*') if p.is_file()}
result=dict(parent=str(parent),target=str(target),changes=changes,source_sha256=closure,selection=selection,
    baseline_library_sha256=digest(parent/'build-coalesced-bf16-body/libgarnet.dylib'),
    scope='Private row80 F32 NDArray candidate; bounded strong-owner views with native fallback after256 new entries. No performance or full validation claim.',full_goal_met=False)
(target/'preparation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(target=str(target),copied_source_files=len(closure),changes=changes,full_goal_met=False)))
