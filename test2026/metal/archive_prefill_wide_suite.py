"""Archive completed wide-prefill validation/comparisons and exact sources."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();archive=Path(sys.argv[2]).resolve();scripts=root/'source/test2026/metal'
names=('prefill-wide-build-20261013','prefill-wide-checks-20261013','native-ab-prefill-wide-20261013',
    'native-ab-prefill-wide-summary-20261013','vllm-matched-vl-prefill-wide-20261013','vl-prefill-wide-summary-20261013',
    'vllm-matched-text-prefill-wide-20261013','text-prefill-wide-summary-20261013',
    'native-frame-prefill-wide-20261013','native-frame-prefill-wide-summary-20261013')
for name in names:assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0,name
snapshot=root/'logs/prefill-wide-source-20261013';snapshot.mkdir(exist_ok=False)
files=[root/'source/src/backends/metal/metal_context.mm',root/'build_prefill_wide.py',root/'compare_vllm_before_native_frames.py']
files.extend(scripts/name for name in ('compare_vllm_metal.py','summarize_native_frames.py','make_prefill_tile_shader.py',
    'three_model_suite.py','summarize_native_builds.py','summarize_vllm_comparison.py','summarize_frame_comparison.py',
    'frame_worker_garnet.py','frame_worker_vllm.py','frame_preprocessing.py','run_matched_frame_process.py','run_matched_garnet.py','run_matched_vllm.py'))
manifest={}
for path in files:
    data=path.read_bytes();(snapshot/path.name).write_bytes(data);manifest[path.name]=hashlib.sha256(data).hexdigest()
(snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(archive,'w:gz') as output:
    for folder in [root/'logs'/name for name in names]+[snapshot]:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix not in ('.f32','.rgb'):
                output.add(path,arcname=str(Path(folder.name)/path.relative_to(folder)),recursive=False)
print('Archived completed wide-prefill suite',archive)
