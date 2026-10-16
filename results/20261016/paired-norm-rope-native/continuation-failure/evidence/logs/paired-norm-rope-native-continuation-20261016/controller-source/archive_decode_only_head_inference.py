"""Preserve terminal route-policy gates, trained recovery and complete ABBA."""
from pathlib import Path
source=Path(__file__).with_name('archive_fused_head_argmax_inference.py').read_text()
for old,new in (
    ('fused-head-argmax-checks-20261015','decode-only-head-checks-20261015'),
    ('fused-head-argmax-native-20261015','decode-only-head-native-20261015'),
    ('fused-head-argmax-staging','decode-only-head-staging'),
    ('fused-head-argmax-original-source-20261015','decode-only-head-original-source-20261015'),
    ('build-fused-head-argmax','build-decode-only-head'),
    ('archive_fused_head_argmax_inference.py','archive_decode_only_head_inference.py'),
    ('run_guarded_fused_head_argmax_checks.py','run_guarded_decode_only_head_checks.py'),
    ('launch_fused_head_argmax_checks.py','launch_decode_only_head_checks.py'),
    ('run_fused_head_session_checks.py','run_decode_only_head_session_checks.py'),
    ('compare_fused_head_argmax.py','compare_decode_only_head.py'),
    ('run_fused_head_argmax_profiles.py','run_decode_only_head_profiles.py')):
    source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
