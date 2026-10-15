"""Pin exact prior bytes; preserve isolated candidate source for audit and installation."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/parallel-vision-staging';stage.mkdir(exist_ok=False)
paths=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/metal_context.h','src/backends/metal/compiled_graph.cpp','src/backends/metal/compiled_graph.h','src/entry/metal_main.cpp','test2026/metal/frame_worker_garnet.py'];original={};manifest={}
extra=['src/backends/metal/parallel_vision_context_shader.h','test2026/metal/run_parallel_vision_grid.py','test2026/metal/test_parallel_vision_grid.py','test2026/metal/run_guarded_parallel_vision_checks.py','test2026/metal/launch_parallel_vision_checks.py']
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
    for name in paths+extra:
        if name in paths:original[name]=hashlib.sha256(subprocess.check_output(['git','show','00f524e9:'+name],cwd=root)).hexdigest()
        raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_text(json.dumps(value,indent=2)+'\n')
print('Prepared',len(manifest),'pinned candidate files')
