"""All six original profiles; sequential ABBA and complete output matching."""
from pathlib import Path
source=Path(__file__).with_name('run_vision_global_alpha_profiles.py').read_text()
source=source.replace('compare_vision_global_alpha.py','compare_mlp_prefill_native.py')
source=source.replace("confirm='--confirm' in sys.argv[3:]",'confirm=False')
exec(compile(source,__file__,'exec'))
