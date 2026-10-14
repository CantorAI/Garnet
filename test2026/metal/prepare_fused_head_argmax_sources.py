"""Pin current originals and complete candidate source for exact installation."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/fused-head-argmax-staging';stage.mkdir(exist_ok=False)
paths=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/compiled_graph.cpp','src/backends/metal/compiled_graph.h','src/backends/metal/decoder_session.h','src/backends/metal/decoder_session.cpp','src/entry/metal_main.cpp','src/backends/metal/fused_head_argmax_shader.h','test2026/metal/fused_head_argmax_test.cpp','test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py']
new={'src/backends/metal/fused_head_argmax_shader.h','test2026/metal/fused_head_argmax_test.cpp'};original={};candidate={}
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
    for name in paths:
        if name not in new:original[name]=hashlib.sha256(subprocess.check_output(['git','show','4d1f807c:'+name],cwd=root)).hexdigest()
        raw=(root/name).read_bytes();candidate[name]=hashlib.sha256(raw).hexdigest();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',candidate)]:
    (stage/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
(stage/'install_fused_head_argmax_sources.py').write_bytes((root/'test2026/metal/install_fused_head_argmax_sources.py').read_bytes())
print('Prepared',len(paths),'candidate sources; pinned',len(original),'unchanged remote originals')
