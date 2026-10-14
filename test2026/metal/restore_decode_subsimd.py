"""Restore only the verified authored integration files in the owned Mac root."""
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();assert r==Path('/Users/zsdx/garnet-metal-investigation')
s=(r/'source').resolve();backup=r/'logs/decode-subsimd-original-source-20261013'
manifest=json.loads((r/'work/decode-subsimd-manifest.json').read_text())
names=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp',
       'test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py']
for job in ('decode-subsimd-checks-20261013','decode-subsimd-inference-20261013'):
    assert json.loads((r/'logs'/job/'exit.json').read_text())['exit_code']==0
pending=[]
for name in names:
    target=(s/name).resolve();assert target.is_relative_to(s)
    item=manifest[name];raw=(backup/name).read_bytes()
    assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256'],name
    assert hashlib.sha256(raw).hexdigest()==item['before_sha256'],name
    pending.append((name,target,raw))
result={}
for name,target,raw in pending:
    target.write_bytes(raw);result[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    assert result[name]==manifest[name]['before_sha256']
(r/'work/decode-subsimd-restored.json').write_text(json.dumps(result,indent=2)+'\n')
print('Restored',len(result),'verified integration files; evidence/builds preserved')
