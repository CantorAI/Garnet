"""Require recovery gates, then all six fresh repaired-reset profiles."""
from pathlib import Path
source=Path(__file__).with_name('run_dirty_prefix_reset_profiles.py').read_text().replace('run_guarded_prefix_reset_session_tests.py','run_guarded_prefix_reset_repair_session_tests.py').replace('compare_dirty_prefix_reset.py','compare_dirty_prefix_reset_repair.py')
exec(compile(source,__file__,'exec'))
