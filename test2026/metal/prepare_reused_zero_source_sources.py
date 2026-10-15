"""Pin retained-source-only reset changes against the archived deferred source."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/reused-zero-source-staging';stage.mkdir(exist_ok=False)
paths=['cmake/GarnetMetal.cmake','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','src/entry/metal_main.cpp','test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py'];original={};manifest={}
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
    for name in paths:
        original[name]=hashlib.sha256(subprocess.check_output(['git','show','3bf9bfe5:'+name],cwd=root)).hexdigest();raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
print('Prepared and pinned',len(paths),'retained-source candidate files')
