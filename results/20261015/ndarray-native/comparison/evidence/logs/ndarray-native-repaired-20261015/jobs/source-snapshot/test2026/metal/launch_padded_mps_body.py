"""Require completed native trials before the exclusive projection experiment."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for trial in ('screened-prefill-reset-native-20261015','screened-prefill-reset-confirm-native-20261015'):assert json.loads((root/'logs'/trial/'exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_padded_mps_body.py')
exec(compile(source,__file__,'exec'))
