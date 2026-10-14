"""Reject rollback if current files differ from the authored installed source."""
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();manifest=json.loads((r/'results/20261013/decode-subsimd/native-source-manifest.json').read_text())
for name in ('cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp',
             'test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py'):
    raw=(r/name).read_bytes().replace(b'\r\n',b'\n');assert hashlib.sha256(raw).hexdigest()==manifest[name]['sha256'],name
print('Five authored files match candidate manifest')
