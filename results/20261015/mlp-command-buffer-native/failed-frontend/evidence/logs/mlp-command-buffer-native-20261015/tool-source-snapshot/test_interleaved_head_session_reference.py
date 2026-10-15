"""Keep frozen session fixtures while observing actual index-layout dispatches."""
from pathlib import Path
source=Path(__file__).with_name('test_prefix_reset_session_reference.py').read_text().replace('run_prefix_reset_session_reference.py','run_interleaved_head_session_reference.py')
exec(compile(source,__file__,'exec'))
