"""All frozen native profiles with complete generated outputs and actual seed rows."""
from pathlib import Path
source=Path(__file__).with_name('run_serial_vision_profiles.py').read_text().replace('compare_serial_vision.py','compare_small_seed.py')
exec(compile(source,__file__,'exec'))
