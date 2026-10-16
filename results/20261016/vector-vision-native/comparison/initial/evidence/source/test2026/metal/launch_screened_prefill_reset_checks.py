"""Begin isolated combination checks only after the fresh comparator is terminal."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/screened-head-vllm-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/screened-head-vllm-verification.json').read_text())
assert audit['actual_dtype_audits']==3 and audit['complete_matched_outputs']==72 and audit['profiles']==6
manifest=json.loads((root/'screened-head-staging/manifest.json').read_text())
assert all(hashlib.sha256((root/'source'/n).read_bytes()).hexdigest()==sha for n,sha in manifest.items())
assert hashlib.sha256((root/'build-screened-head/libgarnet.dylib').read_bytes()).hexdigest()=='2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23'
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_guarded_screened_prefill_reset_checks.py')
exec(compile(source,__file__,'exec'))
