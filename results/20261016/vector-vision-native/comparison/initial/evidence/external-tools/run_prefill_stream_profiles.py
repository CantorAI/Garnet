"""All six sequential native ABBA profiles after complete model validation."""
from pathlib import Path
source=Path(__file__).with_name('run_vision_global_alpha_profiles.py').read_text().replace('compare_vision_global_alpha.py','compare_prefill_stream.py')
exec(compile(source,__file__,'exec'))
