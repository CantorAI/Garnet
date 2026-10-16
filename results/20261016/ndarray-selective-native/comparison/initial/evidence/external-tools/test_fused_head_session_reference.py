"""Persistent trained fixtures for greedy-only fusion, fallback and recovery."""
from pathlib import Path
source=Path(__file__).with_name('test_prefix_reset_session_reference.py').read_text().replace('run_prefix_reset_session_reference.py','run_fused_head_session_reference.py')
exec(compile(source,__file__,'exec'))
