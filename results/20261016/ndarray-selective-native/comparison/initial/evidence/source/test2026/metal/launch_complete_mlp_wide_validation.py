"""Quiet execution of one omitted stage; original27 archive and source stay frozen."""
from pathlib import Path
import json,sys

root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/mlp-prefill-native-checks-resume-20261015/exit.json').read_text())['exit_code']==0
assert json.loads((root/'logs/mlp-prefill-native-finalize-20261015/exit.json').read_text())['exit_code']==1
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','complete_mlp_wide_validation.py')
exec(compile(source,__file__,'exec'))
