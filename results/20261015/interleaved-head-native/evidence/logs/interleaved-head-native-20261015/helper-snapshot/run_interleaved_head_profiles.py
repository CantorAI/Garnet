"""All unchanged model/batch profiles, complete outputs and actual layout counts."""
from pathlib import Path
source=Path(__file__).with_name('run_small_seed_profiles.py').read_text().replace('compare_small_seed.py','compare_interleaved_head.py')
exec(compile(source,__file__,'exec'))
