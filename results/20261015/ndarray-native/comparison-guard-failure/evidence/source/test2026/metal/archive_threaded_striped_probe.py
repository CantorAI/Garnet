"""Archive terminal decode/resize composition trials, source and dependency proof."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();files={};jobs={}
for folder in sorted((root/'logs').glob('threaded-striped-probe-*20261015')):
    jobs[folder.name]=json.loads((folder/'exit.json').read_text())
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix in ('.py','.json','.log','.txt'):files['evidence/'+str(p.relative_to(root))]=p
names=('frame_threaded_striped.py','frame_preprocessing_striped.py','frame_webp_threaded.py','frame_frontend_candidates.py','frame_preprocessing.py','test_threaded_striped_exact.py','test_webp_thread_exact.py','run_threaded_striped_probe.py','run_compact_frame_probe.py','queue_threaded_striped_probe.py','archive_threaded_striped_probe.py','verify_threaded_striped_probe.py','compare_vllm_metal.py','webp_thread_decode.cpp')
for name in names:files['evidence/source/'+name]=root/'source/test2026/metal'/name
assert jobs;library=root/'logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib'
proof=dict(files={},jobs=jobs,library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
assert proof['library_sha256']=='3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3'
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),jobs=jobs,sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
