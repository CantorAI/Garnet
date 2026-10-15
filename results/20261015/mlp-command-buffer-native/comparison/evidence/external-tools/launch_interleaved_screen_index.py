"""Require terminal archived native work before the separate index-layout probe."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('small-seed-checks-20261015','small-seed-native-20261015','small-seed-native-confirm-20261015'):assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/small-seed-native-verification.json').read_text());assert audit['full_output_requests']==72 and audit['actual_candidate_seed_rows']==1024
confirmation=json.loads((root/'work/small-seed-native-confirm-verification.json').read_text());assert confirmation['full_output_requests']==72 and confirmation['actual_candidate_seed_rows']==1024
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_interleaved_screen_index.py')
exec(compile(source,__file__,'exec'))
