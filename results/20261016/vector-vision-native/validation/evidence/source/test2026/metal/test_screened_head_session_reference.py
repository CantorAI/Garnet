"""Persist trained B1/B2 fixtures for screened sessions without serving timing."""
from pathlib import Path
source=Path(__file__).with_name('test_prefix_reset_session_reference.py').read_text().replace('run_prefix_reset_session_reference.py','run_screened_head_session_reference.py')
exec(compile(source,__file__,'exec'))
