"""Wait on the exact owned deferred-clear gate before session tests and ABBA."""
from pathlib import Path
source=Path(__file__).with_name('queue_fused_head_argmax_profiles.py').read_text()
source=source.replace('fused-head-argmax-checks-20261015','deferred-state-zero-checks-20261015').replace('pid=68935','pid=26187').replace('launch_fused_head_argmax_checks.py','launch_deferred_state_zero_checks.py').replace('fused-head-argmax-staging/manifest.json','deferred-state-zero-staging/manifest.json')
source=source.replace('queue_fused_head_argmax_profiles.py','queue_deferred_state_zero_profiles.py').replace('run_fused_head_session_checks.py','run_deferred_zero_session_checks.py').replace('test_fused_head_session_reference.py','test_deferred_zero_session_reference.py').replace('run_fused_head_session_reference.py','run_deferred_zero_session_reference.py').replace('run_fused_head_argmax_profiles.py','run_deferred_state_zero_profiles.py').replace('compare_fused_head_argmax.py','compare_deferred_state_zero.py')
exec(compile(source,__file__,'exec'))
