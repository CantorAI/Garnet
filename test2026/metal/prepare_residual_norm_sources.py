"""Prepare candidate archive and pinned prior identities from the branch base."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/residual-norm-staging';stage.mkdir(exist_ok=False)
paths=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/compiled_graph.cpp','src/backends/metal/decoder_session.h','src/backends/metal/decoder_session.cpp','src/entry/metal_main.cpp','test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py','test2026/metal/matmul_test.cpp']
original={};manifest={}
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
 for name in paths:
  prior=subprocess.check_output(['git','show','8157a4d5:'+name],cwd=root);original[name]=hashlib.sha256(prior).hexdigest()
  raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
(stage/'original.json').write_bytes((json.dumps(original,indent=2)+'\n').encode());(stage/'manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
print('Prepared',len(paths),'candidate source files')
