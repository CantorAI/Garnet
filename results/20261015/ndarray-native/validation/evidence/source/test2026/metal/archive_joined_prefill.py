"""Archive joined-prefill integration checks and actual sequential model results."""
import hashlib,json,platform,subprocess,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();entries=[];omitted=[];jobs={}
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
for name in ('joined-prefill-checks-20261013','joined-prefill-native-20261013','joined-prefill-native-confirm-20261013'):
    folder=r/'logs'/name
    if not folder.exists():continue
    assert (folder/'exit.json').is_file(),'Job not terminal: '+name
    jobs[name]=json.loads((folder/'exit.json').read_text())
    for path in sorted(folder.rglob('*')):
        if not path.is_file():continue
        entry=dict(path=str(path.relative_to(r)),bytes=path.stat().st_size,sha256=digest(path))
        if path.suffix in ('.json','.log','.txt'):entries.append((path,entry))
        else:omitted.append(entry)
folder=r/'logs/joined-prefill-original-source-20261013'
for path in sorted(folder.rglob('*')):
    if path.is_file():entries.append((path,dict(path=str(path.relative_to(r)),bytes=path.stat().st_size,sha256=digest(path))))
manifest=json.loads((r/'joined-prefill-source-manifest.json').read_text())
names=list(manifest)+['test2026/metal/install_joined_prefill_candidate.py','test2026/metal/archive_joined_prefill.py','test2026/metal/test_exact_dense_tail_reference.py','test2026/metal/reference_dense.py','test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py','test2026/metal/run_matched_frame_process.py','test2026/metal/compare_vllm_metal.py','test2026/metal/run_after_owned_job.py']
for name in names:
    path=r/'source'/name;entries.append((path,dict(path=str(path.relative_to(r)),bytes=path.stat().st_size,sha256=digest(path))))
for name in ('joined-prefill-source-manifest.json','joined-prefill-original-manifest.json','joined-prefill-source.tar.gz'):
    path=r/name;entries.append((path,dict(path=name,bytes=path.stat().st_size,sha256=digest(path))))
control=r/'build-resident-kv-inplace/libgarnet.dylib';candidate=r/'build-joined-prefill/libgarnet.dylib'
assert digest(control)=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
metadata=dict(scope=__doc__,jobs=jobs,files=[e for _,e in entries],omitted=omitted,control_sha256=digest(control),candidate_sha256=digest(candidate) if candidate.exists() else None,versions=dict(python=sys.version,macos=platform.platform(),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)))
proof=target.with_suffix('.manifest.json');proof.write_text(json.dumps(metadata,indent=2))
with tarfile.open(target,'w:gz') as archive:
    archive.add(proof,arcname='manifest.json')
    for path,e in entries:archive.add(path,arcname=e['path'])
print(json.dumps(dict(archive=str(target),bytes=target.stat().st_size,sha256=digest(target),files=len(entries),candidate_sha256=metadata['candidate_sha256'])))
