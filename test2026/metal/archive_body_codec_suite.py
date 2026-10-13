"""Preserve terminal experimental body-codec evidence before source reversion."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();archive=Path(sys.argv[2]).resolve();scripts=root/'source/test2026/metal'
names=('body-codec-build-20261013','body-codec-checks-20261013','native-ab-body-codec-20261013',
    'native-ab-body-codec-summary-20261013','native-frame-body-codec-20261013','native-frame-body-codec-summary-20261013')
for name in names:assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0,name
snapshot=root/'logs/body-codec-source-20261013';snapshot.mkdir(exist_ok=False)
paths=[root/'source/src/backends/metal/metal_context.mm',root/'source/cmake/GarnetMetal.cmake',root/'build_body_codec.py']
paths.extend(scripts/name for name in ('compare_vllm_metal.py','three_model_suite.py','summarize_native_builds.py','summarize_native_frames.py','frame_worker_garnet.py','run_matched_frame_process.py','run_matched_garnet.py'))
manifest={}
for path in paths:
    data=path.read_bytes();(snapshot/path.name).write_bytes(data);manifest[path.name]=hashlib.sha256(data).hexdigest()
(snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(archive,'w:gz') as output:
    for folder in [root/'logs'/name for name in names]+[snapshot]:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix not in ('.f32','.rgb'):output.add(path,arcname=str(Path(folder.name)/path.relative_to(folder)),recursive=False)
print('Archived body-codec suite',archive)
