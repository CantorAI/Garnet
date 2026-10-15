"""Verify the exact running validation PID, then run all six native profiles."""
from pathlib import Path
source=Path(__file__).with_name('queue_vision_global_alpha_profiles.py').read_text().replace('vision-global-alpha-checks-20261014','dirty-prefix-reset-checks-20261014').replace('launch_vision_global_alpha_checks.py','launch_dirty_prefix_reset_checks.py').replace('run_vision_global_alpha_profiles.py','run_dirty_prefix_reset_profiles.py').replace('assert pid==13040','assert pid==44364')
exec(compile(source,str(Path(__file__)),'exec'))
