"""Wait for exact native validation, then trained recovery and complete ABBA."""
from pathlib import Path
source=Path(__file__).with_name('queue_fused_head_argmax_profiles.py').read_text()
source=source.replace('fused-head-argmax-checks-20261015','screened-head-checks-20261015').replace('pid=68935','pid=91274').replace('launch_fused_head_argmax_checks.py','launch_screened_head_checks.py').replace('fused-head-argmax-staging/manifest.json','screened-head-staging/manifest.json').replace('len(stages)==19','len(stages)==20')
for old,new in [('queue_fused_head_argmax_profiles.py','queue_screened_head_profiles.py'),('run_fused_head_session_checks.py','run_screened_head_session_checks.py'),('test_fused_head_session_reference.py','test_screened_head_session_reference.py'),('run_fused_head_session_reference.py','run_screened_head_session_reference.py'),('run_fused_head_argmax_profiles.py','run_screened_head_profiles.py'),('compare_fused_head_argmax.py','compare_screened_head.py')]:source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
