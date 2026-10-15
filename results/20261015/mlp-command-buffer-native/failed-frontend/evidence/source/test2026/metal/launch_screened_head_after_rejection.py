"""Independent head component only, after terminal archived vision rejection."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('parallel-vision-checks-20261015','parallel-vision-native-20261015','screened-head-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==1
audit=json.loads((root/'work/parallel-vision-rejection-verification.json').read_text())
assert audit['original_successful_stages']==19 and audit['native_requests']==0 and audit['worst_tolerance_ratio']>1
assert not (root/'logs/parallel-vision-native-20261015/native').exists()
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_screened_head.py')
exec(compile(source,__file__,'exec'))
