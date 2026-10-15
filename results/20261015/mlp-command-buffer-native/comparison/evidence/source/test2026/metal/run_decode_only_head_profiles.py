"""Six frozen complete-output profiles, with contention retries unchanged."""
from pathlib import Path
source=Path(__file__).with_name('run_vision_global_alpha_profiles.py').read_text().replace('compare_vision_global_alpha.py','compare_decode_only_head.py')
exec(compile(source,__file__,'exec'))
