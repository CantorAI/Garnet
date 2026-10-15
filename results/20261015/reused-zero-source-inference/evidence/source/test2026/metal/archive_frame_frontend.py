"""Archive fresh-frame frontend attribution, exact checks and matched inference."""
import hashlib,json,platform,subprocess,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();entries=[];omitted=[];jobs={}
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
for name in ('frame-frontend-profile-20261013','frame-frontend-profile-retry1-20261013','frame-frontend-abba-20261013','frame-frontend-webp-abba-20261013','frame-frontend-exact-20261013','frame-frontend-exact-fresh-20261013','frame-frontend-selected-exact-20261013','frame-frontend-inference-20261013'):
    folder=r/'logs'/name;assert (folder/'exit.json').is_file(),'Job not terminal: '+name;jobs[name]=json.loads((folder/'exit.json').read_text())
    for path in sorted(folder.rglob('*')):
        if not path.is_file():continue
        entry=dict(path=str(path.relative_to(r)),bytes=path.stat().st_size,sha256=digest(path))
        if path.suffix in ('.json','.log','.txt','.py'):entries.append((path,entry))
        else:omitted.append(entry)
names=('frame_preprocessing.py','frame_preprocessing_control.py','frame_frontend_candidates.py','profile_frame_frontend.py','compare_frame_frontends.py','test_frame_frontend_exact.py','run_matched_frame_process.py','compare_frame_frontend_inference.py','run_frame_frontend_inference.py','archive_frame_frontend.py','frame_worker_garnet.py','frame_worker_vllm.py','compare_vllm_metal.py','benchmark_matched_process.py','run_after_owned_job.py')
for name in names:
    path=r/'source/test2026/metal'/name;entries.append((path,dict(path=str(path.relative_to(r)),bytes=path.stat().st_size,sha256=digest(path))))
control=r/'build-resident-kv-inplace/libgarnet.dylib';assert digest(control)=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
metadata=dict(scope=__doc__,jobs=jobs,files=[e for _,e in entries],omitted=omitted,selected_library_sha256=digest(control),versions=dict(python=sys.version,macos=platform.platform()))
manifest=target.with_suffix('.manifest.json');manifest.write_text(json.dumps(metadata,indent=2))
with tarfile.open(target,'w:gz') as archive:
    archive.add(manifest,arcname='manifest.json')
    for path,e in entries:archive.add(path,arcname=e['path'])
print(json.dumps(dict(archive=str(target),bytes=target.stat().st_size,sha256=digest(target),files=len(entries))))
