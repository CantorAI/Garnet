"""Repeat all six profiles only after the complete combination audit passes."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/screened-prefill-reset-native-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/screened-prefill-reset-verification.json').read_text())
assert audit['original_validation_stages']==19 and audit['additional_trained_head_queries']==51
assert audit['trained_recovery_cases']==audit['prefix_recovery_cases']==6 and audit['full_output_requests']==72
assert audit['all_actual_submission_and_clear_counts_checked'] and len(audit['profiles'])==6
meta=json.loads((root/'logs/screened-prefill-reset-checks-20261015/jobs/metadata.json').read_text())
assert hashlib.sha256((root/'build-screened-prefill-reset/libgarnet.dylib').read_bytes()).hexdigest()==meta['candidate_sha256']
manifest=json.loads((root/'screened-head-staging/manifest.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_screened_prefill_reset_profiles.py')
exec(compile(source,__file__,'exec'))
