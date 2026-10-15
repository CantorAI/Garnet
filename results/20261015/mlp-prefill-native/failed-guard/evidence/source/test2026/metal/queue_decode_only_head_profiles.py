"""Verify the exact live route-policy validation handle before benchmarking."""
from pathlib import Path
source=Path(__file__).with_name('queue_fused_head_argmax_profiles.py').read_text()
source=source.replace('fused-head-argmax-checks-20261015','decode-only-head-checks-20261015').replace('pid=68935','pid=96768').replace('launch_fused_head_argmax_checks.py','launch_decode_only_head_checks.py').replace('fused-head-argmax-staging/manifest.json','decode-only-head-staging/manifest.json')
source=source.replace('queue_fused_head_argmax_profiles.py','queue_decode_only_head_profiles.py').replace('run_fused_head_session_checks.py','run_decode_only_head_session_checks.py').replace('run_fused_head_argmax_profiles.py','run_decode_only_head_profiles.py').replace('compare_fused_head_argmax.py','compare_decode_only_head.py')
exec(compile(source,__file__,'exec'))
