import json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['mps-bf16-compatibility-20261013','exact-half-weight-survey-20261013','mps-exact-half-compatibility-20261013']
with tarfile.open(root/'exact-half-probe-evidence.tar.gz','w:gz') as archive:
 for name in names:
  folder=root/'logs'/name;assert json.loads((folder/'exit.json').read_text())['exit_code']==0
  for path in sorted(folder.rglob('*')):
   if path.is_file():archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['probe_mps_bf16.mm','probe_mps_exact_half.mm','survey_exact_half_weights.py']:
  archive.add(root/'source/test2026/metal'/name,arcname='probe-source/'+name)
 for name in ['run_mps_bf16_probe.py','run_mps_exact_half_probe.py','archive_exact_half_probe.py']:
  archive.add(root/name,arcname='probe-source/'+name)
print('Archived exact-half compatibility and CPU survey evidence')
