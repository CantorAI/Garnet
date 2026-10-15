"""Persist actual trained partial/nonfinite fixtures for body-shape sessions."""
from pathlib import Path
source=Path(__file__).with_name('test_prefix_reset_session_reference.py').read_text().replace('run_prefix_reset_session_reference.py','run_body_shape_session_reference.py')
exec(compile(source,__file__,'exec'))
