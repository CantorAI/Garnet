"""Require quiet resources before the full optional joined-MLP validation."""
from pathlib import Path
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text()
source=source.replace('run_guarded_global_half_alpha_checks.py','run_guarded_joined_global_mlp_checks.py')
exec(compile(source,str(Path(__file__)),'exec'))
