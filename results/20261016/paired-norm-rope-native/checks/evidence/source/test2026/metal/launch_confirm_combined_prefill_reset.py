"""Repeat only after independent full initial audit and terminal component queue."""
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/combined-prefill-reset-20261015/exit.json').read_text())['exit_code']==0
# No inference may overlap the queued component, even if that probe is rejected.
assert (root/'logs/decode-eight-lane-20261015/exit.json').is_file()
audit=json.loads((root/'work/combined-prefill-reset-initial-verification.json').read_text())
assert audit['numerical_stages']==22 and audit['full_output_requests']==72
assert all(audit[k] for k in ('all_actual_prefill_partial_submissions_checked','all_actual_dirty_prefix_reset_bytes_checked','all_three_model_recovery_suites_checked'))
meta=json.loads((root/'logs/combined-prefill-reset-20261015/checks/metadata.json').read_text())
assert hashlib.file_digest((root/'build-combined-prefill-reset/libgarnet.dylib').open('rb'),'sha256').hexdigest()==meta['candidate_sha256']
assert all(hashlib.file_digest((root/'source'/name).open('rb'),'sha256').hexdigest()==sha for name,sha in meta['source_sha256'].items())
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_combined_prefill_reset_profiles.py')
exec(compile(source,__file__,'exec'))
