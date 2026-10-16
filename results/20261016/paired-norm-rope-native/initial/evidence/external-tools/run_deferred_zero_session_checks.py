"""Guard all six trained reset/snapshot/private-state/nonfinite recovery cases."""
from pathlib import Path
source=Path(__file__).with_name('run_fused_head_session_checks.py').read_text().replace('build-fused-head-argmax','build-deferred-state-zero').replace('test_fused_head_session_reference.py','test_deferred_zero_session_reference.py')
exec(compile(source,__file__,'exec'))
