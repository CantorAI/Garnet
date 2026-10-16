"""Require the independently audited initial component before its confirmation."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve();assert json.loads((root/'logs/serial-vision-context-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/serial-vision-context-verification.json').read_text())
assert audit['shape_variant_checks']==120 and audit['failed_admissions']==0 and audit['admitted_profiles']==120
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_serial_vision_context_confirm.py')
exec(compile(source,__file__,'exec'))
