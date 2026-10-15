"""Repeat all native profiles only after independent complete-output audit and quiet resources."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
audit=json.loads((root/'work/coalesced-body-native-verification.json').read_text())
assert audit['full_output_requests']==72 and audit['actual_coalesced'] and audit['coalesced_threads']==64
assert audit['wide_reference_profiles']==4 and audit['wide_carried_steps']==12 and audit['worst_wide_original_bound_ratio']<=1
for name in ('coalesced-body-native-checks-20261015','coalesced-body-native-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
# This separate profiling component must be terminal; numerical rejection or
# compile failure does not invalidate the already-audited native comparison.
assert isinstance(json.loads((root/'logs/combined-graph-prefill-20261015/exit.json').read_text())['exit_code'],int)
closure=json.loads((root/'logs/coalesced-body-native-20261015/source-identities.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in closure.items())
assert hashlib.sha256((root/'build-coalesced-bf16-body/libgarnet.dylib').read_bytes()).hexdigest()=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
assert hashlib.sha256((root/'build-wide-batch-prefill/libgarnet.dylib').read_bytes()).hexdigest()=='9ee337d9db7d14bcff29a90970c4b5ac9021aee9f89a38a7f6f364051927d77b'
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text()
source=source.replace('run_guarded_global_half_alpha_checks.py','run_coalesced_body_native_profiles.py')
exec(compile(source,__file__,'exec'))
