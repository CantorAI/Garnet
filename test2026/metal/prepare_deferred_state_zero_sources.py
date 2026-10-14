"""Freeze the optional reset against the fully archived route-policy source."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/deferred-state-zero-staging';stage.mkdir(exist_ok=False)
paths=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','src/entry/metal_main.cpp','xModel/qwen35/vl_0_8b/metal_decoder.py','test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py','test2026/metal/deferred_state_zero_test.cpp'];original={};manifest={}
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
    for name in paths:
        if name!='test2026/metal/deferred_state_zero_test.cpp':original[name]=hashlib.sha256(subprocess.check_output(['git','show','c723ecda:'+name],cwd=root)).hexdigest()
        raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
print('Prepared',len(paths),'candidate sources and',len(original),'pinned originals')
