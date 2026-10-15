"""All six unchanged workloads, full-output checks and exclusive retries."""
from pathlib import Path
source=Path(__file__).with_name('run_vision_global_alpha_profiles.py').read_text().replace('compare_vision_global_alpha.py','compare_body_shape.py')
exec(compile(source,__file__,'exec'))
