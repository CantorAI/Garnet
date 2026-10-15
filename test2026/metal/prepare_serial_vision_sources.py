"""Pin existing bytes and the optional strict original-order vision route."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/serial-vision-staging';stage.mkdir(exist_ok=False)
existing=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/metal_context.h','src/backends/metal/compiled_graph.cpp','src/backends/metal/compiled_graph.h','src/entry/metal_main.cpp','test2026/metal/frame_worker_garnet.py']
extra=['src/backends/metal/serial_vision_prefetch_shader.h','test2026/metal/run_serial_vision_grid.py','test2026/metal/test_serial_vision_grid.py','test2026/metal/run_guarded_serial_vision_checks.py','test2026/metal/launch_serial_vision_checks.py']
original={};manifest={}
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
    for name in existing+extra:
        if name in existing:original[name]=hashlib.sha256(subprocess.check_output(['git','show','d788d82c:'+name],cwd=root)).hexdigest()
        raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();entry=tarfile.TarInfo(name);entry.size=len(raw);archive.addfile(entry,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
print('Prepared',len(manifest),'serial vision source identities')
