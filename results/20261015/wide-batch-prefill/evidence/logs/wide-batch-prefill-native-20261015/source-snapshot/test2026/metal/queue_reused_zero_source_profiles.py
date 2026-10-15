"""Wait for exact owned retained-source gates, then lifecycle checks and ABBA."""
from pathlib import Path
source=Path(__file__).with_name('queue_fused_head_argmax_profiles.py').read_text()
source=source.replace('fused-head-argmax-checks-20261015','reused-zero-source-checks-20261015').replace('pid=68935','pid=55933').replace('launch_fused_head_argmax_checks.py','launch_reused_zero_source_checks.py').replace('fused-head-argmax-staging/manifest.json','reused-zero-source-staging/manifest.json')
source=source.replace('queue_fused_head_argmax_profiles.py','queue_reused_zero_source_profiles.py').replace('run_fused_head_session_checks.py','run_reused_zero_session_checks.py').replace('test_fused_head_session_reference.py','test_reused_zero_session_reference.py').replace('run_fused_head_session_reference.py','run_reused_zero_session_reference.py').replace('run_fused_head_argmax_profiles.py','run_reused_zero_source_profiles.py').replace('compare_fused_head_argmax.py','compare_reused_zero_source.py')
exec(compile(source,__file__,'exec'))
