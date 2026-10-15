"""Pin native layout changes against the last measured small-seed source base."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/interleaved-head-staging';stage.mkdir(exist_ok=False)
existing=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/metal_context.h','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','src/backends/metal/screened_head_index.h','src/backends/metal/screened_head_native_impl.h','src/backends/metal/screened_head_shader.h','src/entry/metal_main.cpp','test2026/metal/frame_worker_garnet.py','test2026/metal/run_matched_garnet.py','test2026/metal/screened_head_real_test.mm','test2026/metal/screened_head_test.cpp','test2026/metal/make_screened_head_header.py']
extra=['test2026/metal/run_interleaved_head_session_reference.py','test2026/metal/test_interleaved_head_session_reference.py','test2026/metal/run_guarded_interleaved_head_checks.py','test2026/metal/launch_interleaved_head_checks.py']
original={};manifest={}
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
    for name in existing+extra:
        if name in existing:original[name]=hashlib.sha256(subprocess.check_output(['git','show','b0b0e134cd3e0668e553e84964d4d7e93212f703:'+name],cwd=root)).hexdigest()
        raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();entry=tarfile.TarInfo(name);entry.size=len(raw);archive.addfile(entry,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
print('Prepared',len(manifest),'interleaved native source identities')
