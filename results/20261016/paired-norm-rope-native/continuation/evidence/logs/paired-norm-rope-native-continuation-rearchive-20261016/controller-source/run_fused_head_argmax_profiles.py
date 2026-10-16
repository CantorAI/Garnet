"""Fresh guard/retry orchestration with unchanged complete matched cases."""
from pathlib import Path
source=Path(__file__).with_name('run_vision_global_alpha_profiles.py').read_text().replace('compare_vision_global_alpha.py','compare_fused_head_argmax.py').replace('Vision common alpha','Fused head argmax')
exec(compile(source,__file__,'exec'))
