"""Retain decoder codec measurements and exact code, omit derived buffers."""
import hashlib,json,sys,tarfile
from pathlib import Path
folder=Path(sys.argv[1]).resolve();scripts=Path(sys.argv[2]).resolve();archive=Path(sys.argv[3]).resolve()
assert json.loads((folder/'exit.json').read_text())['exit_code']==0
metadata=json.loads((folder/'jobs/metadata.json').read_text());snapshot=folder/'source-snapshot';snapshot.mkdir(exist_ok=False)
for name,expected in metadata['source_sha256'].items():
    path=scripts.parents[1]/'src/backends/metal/metal_context.mm' if name=='metal_context.mm' else scripts/name
    data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==expected;(snapshot/name).write_bytes(data)
with tarfile.open(archive,'w:gz') as output:
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.suffix!='.bin' and path.name!='lossless_gpu_probe':output.add(path,arcname=str(Path(folder.name)/path.relative_to(folder)),recursive=False)
print('Archived decoder codec evidence',archive)
