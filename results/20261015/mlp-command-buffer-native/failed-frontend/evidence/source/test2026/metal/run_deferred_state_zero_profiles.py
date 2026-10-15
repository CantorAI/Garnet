"""All six complete-output profiles with existing exclusive contention retries."""
from pathlib import Path
source=Path(__file__).with_name('run_vision_global_alpha_profiles.py').read_text().replace('compare_vision_global_alpha.py','compare_deferred_state_zero.py')
exec(compile(source,__file__,'exec'))
