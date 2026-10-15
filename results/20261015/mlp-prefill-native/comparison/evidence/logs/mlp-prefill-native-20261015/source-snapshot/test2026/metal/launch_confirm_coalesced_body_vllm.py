"""Repeat the exact candidate cohort after its first independent matched audit."""
from pathlib import Path
import hashlib,json,sys

root=Path(sys.argv[1]).resolve();prior=root/'logs/coalesced-body-vllm-20261015'
assert json.loads((prior/'pid.json').read_text())['pid']==23200
assert json.loads((prior/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/coalesced-body-vllm-verification.json').read_text())
assert audit['complete_matched_outputs']==72 and audit['actual_dtype_audits']==3 and audit['profiles']==6
# This unrelated diagnostic must be terminal before another inference cohort.
status=json.loads((root/'logs/combined-graph-error-diagnostic-20261015/exit.json').read_text())
assert isinstance(status['exit_code'],int)
selection=json.loads((root/'work/coalesced-body-next-comparator-selection.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in selection['source_manifest'].items())
assert all(hashlib.sha256((root/item['directory']/'libgarnet.dylib').read_bytes()).hexdigest()==item['sha256'] for item in selection['builds'].values())
assert {v['sha256'] for v in selection['builds'].values()}=={'c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'}
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_coalesced_body_vllm.py')
exec(compile(source,__file__,'exec'))
