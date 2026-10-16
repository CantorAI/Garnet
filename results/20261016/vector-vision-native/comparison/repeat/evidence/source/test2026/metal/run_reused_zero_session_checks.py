"""Guard all six trained session lifecycle and recovery cases."""
from pathlib import Path
source=Path(__file__).with_name('run_fused_head_session_checks.py').read_text().replace('build-fused-head-argmax','build-reused-zero-source').replace('test_fused_head_session_reference.py','test_reused_zero_session_reference.py').replace('All three fused-head trained session/fallback/recovery suites PASS','All three retained-source trained session/lifecycle/recovery suites PASS')
exec(compile(source,__file__,'exec'))
