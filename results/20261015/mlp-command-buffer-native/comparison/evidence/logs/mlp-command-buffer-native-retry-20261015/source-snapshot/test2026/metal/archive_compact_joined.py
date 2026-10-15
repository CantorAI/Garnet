"""Archive compact-joined integration checks and actual sequential model results."""
import hashlib,json,platform,subprocess,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();entries=[];omitted=[];jobs={}
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
for name in ('compact-joined-checks-20261013','compact-joined-native-20261013','compact-joined-native-confirm-20261013'):
    folder=r/'logs'/name
    if not folder.exists():continue
    assert (folder/'exit.json').is_file(),'Job not terminal: '+name
    jobs[name]=json.loads((folder/'exit.json').read_text())
    for path in sorted(folder.rglob('*')):
        if not path.is_file():continue
        entry=dict(path=str(path.relative_to(r)),bytes=path.stat().st_size,sha256=digest(path))
        if path.suffix in ('.json','.log','.txt'):entries.append((path,entry))
        else:omitted.append(entry)
folder=r/'logs/compact-joined-original-source-20261013'
for path in sorted(folder.rglob('*')):
    if path.is_file():entries.append((path,dict(path=str(path.relative_to(r)),bytes=path.stat().st_size,sha256=digest(path))))
manifest=json.loads((r/'compact-joined-source-manifest.json').read_text())
names=list(manifest)+['test2026/metal/install_compact_joined.py','test2026/metal/archive_compact_joined.py','test2026/metal/test_exact_dense_tail_reference.py','test2026/metal/reference_dense.py','test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py','test2026/metal/run_matched_frame_process.py','test2026/metal/compare_vllm_metal.py','test2026/metal/compare_compact_joined.py','test2026/metal/run_compact_joined_profiles.py','test2026/metal/verify_compact_joined.py']
for name in names:
    path=r/'source'/name;entries.append((path,dict(path=str(path.relative_to(r)),bytes=path.stat().st_size,sha256=digest(path))))
for name in ('compact-joined-source-manifest.json','compact-joined-original-manifest.json','compact-joined-source.tar.gz'):
    path=r/name;entries.append((path,dict(path=name,bytes=path.stat().st_size,sha256=digest(path))))
control=r/'build-joined-prefill/libgarnet.dylib';candidate=r/'build-compact-joined-prefill/libgarnet.dylib'
assert digest(control)=='cc41fde1f3905fcfb02fed6a50ab0f9d68933e94358352b92db4e44cb5f43dfe'
metadata=dict(scope=__doc__,jobs=jobs,files=[e for _,e in entries],omitted=omitted,control_sha256=digest(control),candidate_sha256=digest(candidate) if candidate.exists() else None,versions=dict(python=sys.version,macos=platform.platform(),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)))
proof=target.with_suffix('.manifest.json');proof.write_text(json.dumps(metadata,indent=2))
with tarfile.open(target,'w:gz') as archive:
    archive.add(proof,arcname='manifest.json')
    for path,e in entries:archive.add(path,arcname=e['path'])
print(json.dumps(dict(archive=str(target),bytes=target.stat().st_size,sha256=digest(target),files=len(entries),candidate_sha256=metadata['candidate_sha256'])))
