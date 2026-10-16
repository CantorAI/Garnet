"""Require terminal archived layout inference and repeat before prefill retuning."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('interleaved-head-checks-20261015','interleaved-head-native-20261015','interleaved-head-native-confirm-20261015'):assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
for name in ('interleaved-head-native-verification.json','interleaved-head-native-confirm-verification.json'):
    audit=json.loads((root/'work'/name).read_text());assert audit['actual_interleaved'] and audit['full_output_requests']==72
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_replanned_vl_prefill.py')
exec(compile(source,__file__,'exec'))
