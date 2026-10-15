"""Require independent strict component/fallback audit before native validation."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();assert json.loads((root/'logs/screened-head-guarded-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/screened-head-guarded-verification.json').read_text())
assert len(audit['unsafe_input_fallback_checks'])==33 and len([p for p in audit['profiles'] if p.get('strict_screen_math')])==51
manifest=json.loads((root/'screened-head-staging/manifest.json').read_text())
assert all(hashlib.sha256((root/'source'/n).read_bytes()).hexdigest()==sha for n,sha in manifest.items())
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_guarded_screened_head_checks.py')
exec(compile(source,__file__,'exec'))
