"""Guard six trained eager-cache, head-exclusion and recovery cases."""
from pathlib import Path
source=Path(__file__).with_name('run_fused_head_session_checks.py').read_text().replace('build-fused-head-argmax','build-body-shape').replace('test_fused_head_session_reference.py','test_body_shape_session_reference.py').replace('All three fused-head trained session/fallback/recovery suites PASS','All three body-shape eager-cache/session/recovery suites PASS')
exec(compile(source,__file__,'exec'))
