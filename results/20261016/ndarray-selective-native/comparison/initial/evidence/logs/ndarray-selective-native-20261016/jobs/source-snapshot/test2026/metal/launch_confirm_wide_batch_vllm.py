"""Repeat the exact matched cohort after its first complete independent audit."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();prior=root/'logs/wide-batch-vllm-20261015'
assert json.loads((prior/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/wide-batch-vllm-verification.json').read_text());assert audit['complete_matched_outputs']==72 and audit['actual_dtype_audits']==3 and audit['profiles']==6
selection=json.loads((root/'work/wide-batch-next-comparator-selection.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in selection['source_manifest'].items())
assert all(hashlib.sha256((root/item['directory']/'libgarnet.dylib').read_bytes()).hexdigest()==item['sha256'] for item in selection['builds'].values())
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_wide_batch_vllm.py')
exec(compile(source,__file__,'exec'))
