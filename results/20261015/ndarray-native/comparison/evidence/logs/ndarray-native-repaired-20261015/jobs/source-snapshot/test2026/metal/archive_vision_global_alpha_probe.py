"""Freeze terminal vision-matrix trials with canonical checkpoint identities."""
from pathlib import Path
source=Path(__file__).with_name('archive_global_half_alpha_projection.py').read_text()
source=source.replace('global-half-alpha-projections*-20261014','vision-global-alpha-probe*-20261014')
source=source.replace('global_half_alpha_gpu.mm','vision_global_alpha_gpu.mm').replace('run_global_half_alpha_projection.py','run_vision_global_alpha_probe.py').replace('launch_global_half_alpha_projection.py','launch_vision_global_alpha_probe.py').replace('archive_global_half_alpha_projection.py','archive_vision_global_alpha_probe.py')
source=source.replace("status=json.loads((folder/'exit.json').read_text());jobs[folder.name]=status","status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==0;jobs[folder.name]=status")
source=source.replace('manifest={}',"files['evidence/source/archive_global_half_alpha_projection.py']=root/'source/test2026/metal/archive_global_half_alpha_projection.py'\nmanifest={}")
exec(compile(source,str(Path(__file__)),'exec'))
