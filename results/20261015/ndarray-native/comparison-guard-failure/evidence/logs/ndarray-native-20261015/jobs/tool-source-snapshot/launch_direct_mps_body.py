"""Do not overlap the previous padded-row experiment or any inference worker."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve();assert json.loads((root/'logs/padded-mps-body-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_direct_mps_body.py')
exec(compile(source,__file__,'exec'))
