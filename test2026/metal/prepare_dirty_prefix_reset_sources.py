"""Pin optional native prefix reset against the completed vision source baseline."""
import hashlib,io,json,subprocess,tarfile,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/dirty-prefix-reset-staging';stage.mkdir(exist_ok='--refresh' in sys.argv[1:])
paths=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/compiled_graph.h','src/backends/metal/compiled_graph.cpp','src/backends/metal/decoder_session.h','src/backends/metal/decoder_session.cpp','src/entry/metal_main.cpp','test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py','test2026/metal/run_resident_kv_reference.py','test2026/metal/dirty_prefix_reset_test.cpp']
original={};manifest={}
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
 for name in paths:
  if name!='test2026/metal/dirty_prefix_reset_test.cpp':original[name]=hashlib.sha256(subprocess.check_output(['git','show','4053c7d4:'+name],cwd=root)).hexdigest()
  raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
for name in ('install_dirty_prefix_reset_sources.py','install_residual_norm_sources.py'):(stage/name).write_bytes((root/'test2026/metal'/name).read_bytes())
print('Prepared',len(paths),'candidate sources and pinned',len(original),'originals')
