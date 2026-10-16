"""Start a streamed component only after the individual-matrix job is terminal."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve();assert json.loads((root/'logs/direct-mps-body-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_streamed_direct_mps_body.py')
exec(compile(source,__file__,'exec'))
