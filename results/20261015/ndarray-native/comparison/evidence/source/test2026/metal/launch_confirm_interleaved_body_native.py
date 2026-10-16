"""Repeat all native profiles after frozen full gates, outputs and component audit."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
audit=json.loads((root/'work/interleaved-body-native-verification.json').read_text())
assert audit['full_output_requests']==72 and audit['wide_reference_profiles']==4
assert audit['wide_carried_steps']==12 and audit['worst_wide_original_bound_ratio']<=1
for name in ('interleaved-body-native-checks-20261015','interleaved-body-native-20261015','packed-body-prefill-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
closure=json.loads((root/'logs/interleaved-body-native-20261015/source-identities.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in closure.items())
meta=json.loads((root/'logs/interleaved-body-native-checks-20261015/jobs/metadata.json').read_text())
assert hashlib.sha256((root/'build-interleaved-bf16-body/libgarnet.dylib').read_bytes()).hexdigest()==meta['candidate_sha256']
assert hashlib.sha256((root/'build-wide-batch-prefill/libgarnet.dylib').read_bytes()).hexdigest()=='9ee337d9db7d14bcff29a90970c4b5ac9021aee9f89a38a7f6f364051927d77b'
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text()
source=source.replace('run_guarded_global_half_alpha_checks.py','run_interleaved_body_native_profiles.py')
exec(compile(source,__file__,'exec'))
