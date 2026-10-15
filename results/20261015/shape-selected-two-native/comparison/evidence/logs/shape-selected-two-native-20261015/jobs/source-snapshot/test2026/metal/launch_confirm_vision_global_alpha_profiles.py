"""Durable launcher for sequential VL B1/B2 confirmation."""
from pathlib import Path
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_confirm_vision_global_alpha_profiles.py')
exec(compile(source,str(Path(__file__)),'exec'))
