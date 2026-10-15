"""Archive terminal CPU/frame jobs; exclude bridge binaries and image arrays."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]);files={};omitted={}
cpu_only='--cpu-only' in sys.argv[3:]
for folder in sorted((r/'logs').glob('webp-thread-*')):
    if cpu_only and 'inference' in folder.name:continue
    assert (folder/'exit.json').is_file(),'Live job: '+str(folder)
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.txt','.log','.py','.cpp'):
            files['evidence/logs/'+str(p.relative_to(r/'logs'))]=p
        elif p.is_file() and p.suffix in ('.dylib','.f32','.rgb'):
            raw=p.read_bytes();omitted[str(p.relative_to(r))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
names=('webp_thread_decode.cpp','frame_webp_threaded.py','run_webp_thread_probe.py','test_webp_thread_exact.py','run_webp_thread_frame_process.py','compare_webp_thread_inference.py','run_webp_thread_inference.py','archive_webp_thread.py','frame_preprocessing.py','frame_frontend_candidates.py','frame_preprocessing_control.py','run_matched_frame_process.py','frame_worker_garnet.py','frame_worker_vllm.py','compare_frame_frontend_inference.py','run_retained_prefill_vllm_profiles.py','compare_vllm_metal.py','audit_comparator_dtypes.py','background_comparison.py','run_after_owned_job.py')
for name in names:files['evidence/source/'+name]=r/'source/test2026/metal'/name
manifest={}
with tarfile.open(destination,'w:gz') as archive:
    for name,path in sorted(files.items()):
        raw=path.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    for name,data in [('evidence/files.json',manifest),('evidence/omitted-proof.json',omitted)]:
        raw=(json.dumps(data,indent=2)+'\n').encode();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(manifest),omitted=len(omitted),archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))
