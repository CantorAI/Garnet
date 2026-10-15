"""Repeat all six profiles only after the initial independent body-route audit."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/body-shape-native-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/body-shape-native-verification.json').read_text())
assert audit['original_validation_stages']==19 and audit['trained_recovery_cases']==6 and audit['full_output_requests']==72 and len(audit['profiles'])==6
assert hashlib.sha256((root/'build-body-shape/libgarnet.dylib').read_bytes()).hexdigest()=='e39bbdc6f731412b003f66aa1efe6f83bc71faecd4eef205dad399a728531d21'
manifest=json.loads((root/'body-shape-staging/manifest.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_body_shape_profiles.py')
exec(compile(source,__file__,'exec'))
