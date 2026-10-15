"""Guard each real-model session/head/fallback/recovery suite exclusively."""
from pathlib import Path
source=Path(__file__).with_name('run_fused_head_session_checks.py').read_text().replace('build-fused-head-argmax','build-screened-head').replace('test_fused_head_session_reference.py','test_screened_head_session_reference.py').replace('fused-head trained','screened-head trained')
exec(compile(source,__file__,'exec'))
