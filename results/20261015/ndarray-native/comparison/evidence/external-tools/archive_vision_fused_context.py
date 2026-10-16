"""Preserve component job logs, generated shaders and their exact source hashes."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]).resolve()
for job in ('vision-fused-context-retry2-20261013','vision-fused-context-confirm-20261013','vision-fused-context-checks-20261013','vision-fused-context-inference-20261013','vision-fused-context-inference-confirm-20261013'):
    assert json.loads((root/'logs'/job/'exit.json').read_text())['exit_code']==0,job
files={}
for name in ('vision-fused-context-20261013','vision-fused-context-retry1-20261013','vision-fused-context-retry2-20261013','vision-fused-context-confirm-20261013','vision-fused-context-checks-20261013','vision-fused-context-inference-20261013','vision-fused-context-inference-confirm-20261013'):
    folder=root/'logs'/name
    if not folder.exists():continue
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.txt','.log','.metal','.py'):
            files['evidence/logs/'+str(path.relative_to(root/'logs'))]=path
for name in ('make_vision_fused_context_shader.py','vision_fused_context_gpu.mm','run_vision_fused_context_probe.py','archive_vision_fused_context.py','compare_vllm_metal.py','background_comparison.py','run_vision_fused_context_checks.py','run_joined_prefill_checks.py','compare_vision_fused_context.py','compare_joined_prefill.py','run_matched_frame_process.py','frame_worker_garnet.py','frame_preprocessing.py'):
    files['evidence/source/'+name]=root/'source/test2026/metal'/name
files['evidence/source/metal_context.mm']=root/'source/src/backends/metal/metal_context.mm'
files['evidence/source/compiled_graph.cpp']=root/'source/src/backends/metal/compiled_graph.cpp'
files['evidence/source/GarnetMetal.cmake']=root/'source/cmake/GarnetMetal.cmake'
for name in ('metal_context.mm','compiled_graph.cpp','GarnetMetal.cmake'):
    files['evidence/original/'+name]=root/'work/vision-fused-context-original'/name
checks=root/'logs/vision-fused-context-checks-20261013/jobs/metadata.json'
if checks.exists():
    for name in json.loads(checks.read_text())['source_sha256']:
        files['evidence/check-source/'+name]=root/'source'/name
for target in ('garnet','garnet_metal'):
    files['evidence/build/'+target+'-flags.make']=root/'build-vision-fused-context/CMakeFiles'/(target+'.dir')/'flags.make'
manifest={}
with tarfile.open(destination,'w:gz') as archive:
    for name,path in files.items():
        data=path.read_bytes();manifest[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());archive.add(path,arcname=name,recursive=False)
    encoded=(json.dumps(manifest,indent=2)+'\n').encode()
    import io
    info=tarfile.TarInfo('evidence/files.json');info.size=len(encoded);archive.addfile(info,io.BytesIO(encoded))
print(json.dumps(dict(files=len(manifest),archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))
