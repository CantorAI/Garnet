import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');folder=root/'logs/native-frame-phases-20261013'
assert json.loads((folder/'exit.json').read_text())['exit_code']==0
metadata=json.loads((folder/'jobs/metadata.json').read_text())
assert hashlib.sha256((root/'source/test2026/metal/frame_worker_garnet.py').read_bytes()).hexdigest()==metadata['worker_sha256']
with tarfile.open(root/'native-frame-phase-evidence.tar.gz','w:gz') as archive:
 for path in sorted(folder.rglob('*')):
  if path.is_file() and path.suffix not in ('.f32','.rgb'):archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['frame_worker_garnet.py','profile_native_frame_phases.py']:
  archive.add(root/'source/test2026/metal'/name,arcname='phase-source/'+name)
 archive.add(root/'archive_native_frame_phases.py',arcname='phase-source/archive_native_frame_phases.py')
print('Archived actual serving phase counters and matching complete frame outputs')
