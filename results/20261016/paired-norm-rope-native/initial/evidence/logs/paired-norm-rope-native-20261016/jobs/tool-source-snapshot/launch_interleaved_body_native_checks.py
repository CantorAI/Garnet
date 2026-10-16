"""Require both matched comparator cohorts archived before native integration."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for name in ('wide-batch-vllm-verification.json','wide-batch-vllm-confirm-verification.json'):
    audit=json.loads((root/'work'/name).read_text());assert audit['complete_matched_outputs']==72 and audit['actual_dtype_audits']==3
for name in ('wide-batch-vllm-20261015','wide-batch-vllm-confirm-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
manifest=json.loads((root/'interleaved-body-native-staging/manifest.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_guarded_interleaved_body_native_checks.py')
exec(compile(source,__file__,'exec'))
