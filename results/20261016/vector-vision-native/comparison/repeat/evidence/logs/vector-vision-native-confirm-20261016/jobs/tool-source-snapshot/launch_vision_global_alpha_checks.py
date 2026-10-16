"""Quiet-gated durable build and all original vision/model validation stages."""
from pathlib import Path
source=Path(__file__).with_name('launch_joined_global_mlp_checks.py').read_text().replace('run_guarded_joined_global_mlp_checks.py','run_guarded_vision_global_alpha_checks.py')
exec(compile(source,str(Path(__file__)),'exec'))
