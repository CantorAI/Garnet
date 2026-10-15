"""Keep the frozen trained session fixtures while checking actual seed counts."""
from pathlib import Path
source=Path(__file__).with_name('test_prefix_reset_session_reference.py').read_text().replace('run_prefix_reset_session_reference.py','run_small_seed_session_reference.py')
exec(compile(source,__file__,'exec'))
