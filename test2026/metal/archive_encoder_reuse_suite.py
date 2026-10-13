"""Retain completed encoder reuse experiment and the custom-abort check."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();archive=Path(sys.argv[2]).resolve();scripts=root/'source/test2026/metal'
names=('encoder-reuse-build-20261013','encoder-reuse-checks-20261013','native-ab-encoder-reuse-20261013',
    'native-ab-encoder-reuse-summary-20261013','native-frame-encoder-reuse-20261013',
    'native-frame-encoder-reuse-summary-20261013','encoder-reuse-abort-check-20261013')
for name in names:assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0,name
snapshot=root/'logs/encoder-reuse-source-20261013';snapshot.mkdir(exist_ok=False)
files=[root/'source/src/backends/metal/metal_context.mm',root/'build_encoder_reuse.py',root/'check_encoder_abort.py']
files.extend(scripts/name for name in ('matmul_test.cpp','compare_vllm_metal.py','three_model_suite.py','summarize_native_builds.py','summarize_native_frames.py','frame_worker_garnet.py','run_matched_frame_process.py','run_matched_garnet.py'))
manifest={}
for path in files:
    data=path.read_bytes();(snapshot/path.name).write_bytes(data);manifest[path.name]=hashlib.sha256(data).hexdigest()
(snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(archive,'w:gz') as output:
    for folder in [root/'logs'/name for name in names]+[snapshot]:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix not in ('.f32','.rgb'):output.add(path,arcname=str(Path(folder.name)/path.relative_to(folder)),recursive=False)
print('Archived terminal encoder reuse evidence',archive)
