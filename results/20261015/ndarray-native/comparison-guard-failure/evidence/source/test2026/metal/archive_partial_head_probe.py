"""Retain terminal bound-probe evidence and exact sources, omit bulky logits."""
import hashlib,json,sys,tarfile
from pathlib import Path
folder=Path(sys.argv[1]).resolve();scripts=Path(sys.argv[2]).resolve();archive=Path(sys.argv[3]).resolve()
json.loads((folder/'exit.json').read_text())
metadata=json.loads((folder/'jobs/metadata.json').read_text());snapshot=folder/'source-snapshot';snapshot.mkdir(exist_ok=False)
for name,expected in metadata['source_sha256'].items():
    source=(scripts/name).read_bytes();assert hashlib.sha256(source).hexdigest()==expected
    (snapshot/name).write_bytes(source)
with tarfile.open(archive,'w:gz') as output:
    for path in sorted(folder.rglob('*')):
        if path.is_file() and not path.name.endswith('.scores.json'):
            output.add(path,arcname=str(Path(folder.name)/path.relative_to(folder)),recursive=False)
print('Archived terminal probe',archive)
