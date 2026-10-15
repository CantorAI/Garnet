"""Archive terminal common-frontend inference studies and exact source identities."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();files={};omitted={};jobs={}
for folder in sorted((root/'logs').glob('striped-frame-inference-*20261015')):
    jobs[folder.name]=json.loads((folder/'exit.json').read_text())
    for p in folder.rglob('*'):
        if not p.is_file():continue
        key=str(p.relative_to(root));raw=p.read_bytes()
        if p.suffix in ('.json','.py','.log','.txt'):files['evidence/'+key]=p
        else:omitted[key]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
names=('frame_preprocessing_striped.py','run_striped_frame_process.py','compare_striped_frame_inference.py','run_striped_frame_inference.py','launch_striped_frame_inference.py','frame_preprocessing.py','frame_preprocessing_control.py','run_matched_frame_process.py','frame_worker_garnet.py','frame_worker_vllm.py','compare_frame_frontend_inference.py','run_webp_thread_inference.py','run_retained_prefill_vllm_profiles.py','compare_vllm_metal.py','audit_comparator_dtypes.py','benchmark_matched_process.py','run_matched_garnet.py','run_matched_vllm.py','background_comparison.py')
for name in names:files['evidence/source/'+name]=root/'source/test2026/metal'/name
assert jobs
proof=dict(files={},jobs=jobs,omitted=omitted,native_library_sha256=hashlib.sha256((root/'build-global-half-alpha/libgarnet.dylib').read_bytes()).hexdigest())
assert proof['native_library_sha256']=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),jobs=jobs,sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
