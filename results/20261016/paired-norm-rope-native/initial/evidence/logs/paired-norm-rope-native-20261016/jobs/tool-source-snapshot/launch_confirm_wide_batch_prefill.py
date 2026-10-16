"""Repeat all six native profiles only after independent complete-output audit."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
audit=json.loads((root/'work/wide-batch-prefill-verification.json').read_text())
assert audit['full_output_requests']==72 and audit['wide_reference_profiles']==4 and audit['wide_carried_steps']==12
body=json.loads((root/'work/interleaved-bf16-body-verification.json').read_text());assert body['numeric_checks']==72 and body['timed_executions']==1152
for name in ('wide-batch-prefill-checks-20261015','wide-batch-prefill-native-20261015','interleaved-bf16-body-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
closure=json.loads((root/'logs/wide-batch-prefill-native-20261015/source-identities.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in closure.items())
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_wide_batch_prefill_profiles.py')
exec(compile(source,__file__,'exec'))
