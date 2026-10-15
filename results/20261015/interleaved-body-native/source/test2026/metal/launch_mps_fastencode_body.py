"""Require the preceding streamed experiment to finish before the option test."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve();assert json.loads((root/'logs/streamed-direct-mps-body-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_mps_fastencode_body.py')
exec(compile(source,__file__,'exec'))
