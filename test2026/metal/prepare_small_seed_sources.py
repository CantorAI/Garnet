"""Pin reviewed native seed changes against the last measured serial-vision base."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/small-seed-staging';stage.mkdir(exist_ok=False)
existing=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/metal_context.h','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','src/backends/metal/screened_head_index.h','src/backends/metal/screened_head_native_impl.h','src/backends/metal/screened_head_shader.h','src/entry/metal_main.cpp','test2026/metal/frame_worker_garnet.py','test2026/metal/run_matched_garnet.py','test2026/metal/screened_head_real_test.mm','test2026/metal/screened_head_test.cpp']
extra=['test2026/metal/make_screened_head_header.py','test2026/metal/run_small_seed_session_reference.py','test2026/metal/test_small_seed_session_reference.py','test2026/metal/run_guarded_small_seed_checks.py','test2026/metal/launch_small_seed_checks.py']
original={};manifest={}
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
    for name in existing+extra:
        if name in existing:original[name]=hashlib.sha256(subprocess.check_output(['git','show','09893ade4d80d39e602000b0e716ff878f82c44d:'+name],cwd=root)).hexdigest()
        raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();entry=tarfile.TarInfo(name);entry.size=len(raw);archive.addfile(entry,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
print('Prepared',len(manifest),'small-seed source identities')
