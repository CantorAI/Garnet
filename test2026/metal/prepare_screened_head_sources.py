"""Pin original bytes before installing the optional native screened-head route."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/screened-head-staging';stage.mkdir(exist_ok=False)
paths=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/metal_context.h','src/backends/metal/compiled_graph.cpp','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','src/entry/metal_main.cpp','test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py']
extra=['src/backends/metal/screened_head_index.h','src/backends/metal/screened_head_shader.h','src/backends/metal/screened_head_native_impl.h','test2026/metal/screened_head_test.cpp','test2026/metal/screened_head_real_test.mm','test2026/metal/test_screened_head_real.py','test2026/metal/run_guarded_screened_head_checks.py','test2026/metal/launch_screened_head_checks.py'];original={};manifest={}
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
    for name in paths+extra:
        if name in paths:original[name]=hashlib.sha256(subprocess.check_output(['git','show','ee2fd469:'+name],cwd=root)).hexdigest()
        raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();entry=tarfile.TarInfo(name);entry.size=len(raw);archive.addfile(entry,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
print('Prepared',len(manifest),'screened-head source identities')
