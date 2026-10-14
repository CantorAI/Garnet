"""Freeze the row-specialized coder head against the completed optional MLP base."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/single-coder-head-staging';stage.mkdir(exist_ok=False)
paths=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/compiled_graph.cpp','src/backends/metal/decoder_session.h','src/backends/metal/decoder_session.cpp','src/entry/metal_main.cpp','test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py','test2026/metal/matmul_test.cpp','xModel/qwen/metal_dense/decoder.py','xModel/qwen35/vl_0_8b/metal_decoder.py','test2026/metal/joined_global_mlp_test.cpp','test2026/metal/single_coder_head_test.cpp']
original={};manifest={}
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
 for name in paths:
  if name!='test2026/metal/single_coder_head_test.cpp':
   prior=subprocess.check_output(['git','show','fc653485:'+name],cwd=root);original[name]=hashlib.sha256(prior).hexdigest()
  raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
for name,data in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_bytes((json.dumps(data,indent=2)+'\n').encode())
for name in ('install_single_coder_head_sources.py','install_residual_norm_sources.py'):(stage/name).write_bytes((root/'test2026/metal'/name).read_bytes())
print('Prepared',len(paths),'sources; pinned',len(original),'originals')
