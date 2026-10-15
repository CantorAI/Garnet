"""Fresh sequential confirmation of text B2 and coder B1 native profiles."""
from pathlib import Path
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text()
source=source.replace('run_global_half_alpha_profiles.py','run_confirm_joined_global_mlp_profiles.py')
exec(compile(source,str(Path(__file__)),'exec'))
