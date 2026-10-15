"""Archive probe evidence/source versions, excluding reproducible weight bytes."""
import json,sys,tarfile
from pathlib import Path
folder=Path(sys.argv[1]).resolve();scripts=Path(sys.argv[2]).resolve();archive=Path(sys.argv[3]).resolve()
status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==0
snapshot=folder/'source-snapshot';snapshot.mkdir(exist_ok=False)
for name in ('probe_lossless_bf16_codec.py','prepare_lossless_gpu_probe.py','run_lossless_gpu_probe.py','lossless_gpu_probe.mm','lossless_gpu_probe.metal'):
    (snapshot/name).write_bytes((scripts/name).read_bytes())
if (scripts/'probe_lossless_nibble_bf16_codec.py').is_file():
    (snapshot/'probe_lossless_nibble_bf16_codec.py').write_bytes((scripts/'probe_lossless_nibble_bf16_codec.py').read_bytes())
with tarfile.open(archive,'w:gz') as output:
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.suffix!='.bin' and path.name!='lossless_gpu_probe':
            output.add(path,arcname=str(Path(folder.name)/path.relative_to(folder)),recursive=False)
print('Saved evidence archive',archive)
