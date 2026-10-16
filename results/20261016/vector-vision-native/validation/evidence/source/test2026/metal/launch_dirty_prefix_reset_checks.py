"""Durable exclusive full-model prefix-reset checks."""
from pathlib import Path
source=Path(__file__).with_name('launch_vision_global_alpha_checks.py').read_text().replace('run_guarded_vision_global_alpha_checks.py','run_guarded_dirty_prefix_reset_checks.py')
exec(compile(source,str(Path(__file__)),'exec'))
