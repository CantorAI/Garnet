"""Pin the failed attempt's exact sources before a declaration-only test repair."""
import hashlib,io,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/single-coder-head-repair-staging';stage.mkdir(exist_ok=False)
original=json.loads((root/'.work/single-coder-head-staging/manifest.json').read_text());candidate={}
with tarfile.open(stage/'source.tar.gz','w:gz') as tar:
 for name in original:
  data=(root/name).read_bytes();candidate[name]=hashlib.sha256(data).hexdigest();info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
assert [n for n in original if original[n]!=candidate[n]]==['test2026/metal/single_coder_head_test.cpp']
for name,data in [('original.json',original),('manifest.json',candidate)]:(stage/name).write_bytes((json.dumps(data,indent=2)+'\n').encode())
for name in ('install_single_coder_head_repair.py','install_residual_norm_sources.py'):(stage/name).write_bytes((root/'test2026/metal'/name).read_bytes())
print('Prepared one declaration repair, preserving all fourteen prior identities')
