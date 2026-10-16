"""Isolate selective serving candidate only after both predeclared component cohorts pass."""
from pathlib import Path
import hashlib,json,shutil,sys
parent=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
assert target.parent==parent and target.name=='ndarray-selective-native-investigation-20261016' and not target.exists()
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
selection=[];identities=None
for trial,sha,auditname in (
    ('ndarray-selective-prefill-20261016','f70a9eb265f99f768c0b23cbf033cadbffb758211febaaae1f5bc887638bf05f','ndarray-selective-prefill-verification.json'),
    ('ndarray-selective-prefill-confirm-20261016','a4de900f6d79249ed4ef69c5ce6c5d51fc09fdcf50f7199ea1f570f4573a62b4','ndarray-selective-prefill-confirm-verification.json')):
    assert digest(parent/(trial+'.tar.gz'))==sha
    assert json.loads((parent/'logs'/trial/'exit.json').read_text())['exit_code']==0
    audit=json.loads((parent/auditname).read_text());assert audit['numeric_checks']==72 and audit['verified_files']==62
    assert not any(v['policy']==2 for v in audit['rejected_profiles'])
    for batch in (1,2):
        profiles={v['policy']:v for v in audit['profiles'] if v['model']=='Qwen3.5-0.8B' and v['batch']==batch}
        assert set(profiles)=={1,2,3}
        for key in ('gpu','wall'):
            assert profiles[2][key+'_fraction']<=.95 and .95<=profiles[3][key+'_fraction']<=1.05
        assert all(c['passed'] and c['actual_ndarray_dispatches']==1 for c in profiles[2]['checks'])
    meta=json.loads((parent/'logs'/trial/'jobs/metadata.json').read_text())
    for name,sha in meta['source_sha256'].items():assert digest(parent/'source'/name)==sha,name
    current={k:meta[k] for k in ('source_sha256','assets','context_generated_sha256','harness_generated_sha256','private_header_sha256','context_macro_defines')}
    if identities is None:identities=current
    else:assert current==identities
    selection.append(dict(trial=trial,archive_sha256=digest(parent/(trial+'.tar.gz')),audit_sha256=digest(parent/auditname)))
assert digest(parent/'build-coalesced-bf16-body/libgarnet.dylib')=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
sys.path.insert(0,str(Path(__file__).parent));import make_ndarray_selective_native as factory
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
    scope='Private no-exact-half F32 NDArray only at physical80/160; 256 strong-owner views/native fallback. No model performance or validation claim.',full_goal_met=False)
(target/'preparation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(target=str(target),copied_source_files=len(closure),changes=changes,full_goal_met=False)))
