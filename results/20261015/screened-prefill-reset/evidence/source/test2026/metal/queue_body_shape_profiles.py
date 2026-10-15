"""Wait for exact owned body-shape validation, then trained recovery and ABBA."""
from pathlib import Path
source=Path(__file__).with_name('queue_fused_head_argmax_profiles.py').read_text()
source=source.replace('fused-head-argmax-checks-20261015','body-shape-checks-20261015').replace('pid=68935','pid=88191').replace('launch_fused_head_argmax_checks.py','launch_body_shape_checks.py').replace('fused-head-argmax-staging/manifest.json','body-shape-staging/manifest.json')
source=source.replace('queue_fused_head_argmax_profiles.py','queue_body_shape_profiles.py').replace('run_fused_head_session_checks.py','run_body_shape_session_checks.py').replace('test_fused_head_session_reference.py','test_body_shape_session_reference.py').replace('run_fused_head_session_reference.py','run_body_shape_session_reference.py').replace('run_fused_head_argmax_profiles.py','run_body_shape_profiles.py').replace('compare_fused_head_argmax.py','compare_body_shape.py')
exec(compile(source,__file__,'exec'))
