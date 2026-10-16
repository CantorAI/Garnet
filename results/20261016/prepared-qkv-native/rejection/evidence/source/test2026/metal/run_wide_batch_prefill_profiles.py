"""Six full native profiles after the original and new wide-batch gates passed."""
from pathlib import Path
source=Path(__file__).with_name('run_vision_global_alpha_profiles.py').read_text().replace('compare_vision_global_alpha.py','compare_wide_batch_prefill.py')
source=source.replace("confirm='--confirm' in sys.argv[3:]",'confirm=False')
exec(compile(source,__file__,'exec'))
