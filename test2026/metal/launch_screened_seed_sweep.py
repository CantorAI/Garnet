"""Require terminal native ABBA and independent archive audit before seed sweep."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('serial-vision-checks-20261015','serial-vision-native-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/serial-vision-native-verification.json').read_text())
assert audit['full_output_requests']==72 and audit['trained_full_grid_profiles']==4
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_screened_seed_sweep.py')
exec(compile(source,__file__,'exec'))
