"""Repeat the promising plan after exact original-bound and full-output audit."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve();assert json.loads((root/'logs/replanned-vl-prefill-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/replanned-vl-prefill-verification.json').read_text());assert audit['independent_reference_profiles']==12 and audit['carried_reference_steps']==36 and audit['full_output_requests']==24
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_confirm_replanned_vl_prefill.py')
exec(compile(source,__file__,'exec'))
