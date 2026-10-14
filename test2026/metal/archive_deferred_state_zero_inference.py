"""Archive all terminal gates, deferred-reset source, recovery and native trials."""
from pathlib import Path
source=Path(__file__).with_name('archive_fused_head_argmax_inference.py').read_text()
for old,new in (
    ('fused-head-argmax-checks-20261015','deferred-state-zero-checks-20261015'),
    ('fused-head-argmax-native-20261015','deferred-state-zero-native-20261015'),
    ('fused-head-argmax-staging','deferred-state-zero-staging'),
    ('fused-head-argmax-original-source-20261015','deferred-state-zero-original-source-20261015'),
    ('build-fused-head-argmax','build-deferred-state-zero'),
    ('archive_fused_head_argmax_inference.py','archive_deferred_state_zero_inference.py'),
    ('run_guarded_fused_head_argmax_checks.py','run_guarded_deferred_state_zero_checks.py'),
    ('launch_fused_head_argmax_checks.py','launch_deferred_state_zero_checks.py'),
    ('run_fused_head_session_checks.py','run_deferred_zero_session_checks.py'),
    ('test_fused_head_session_reference.py','test_deferred_zero_session_reference.py'),
    ('run_fused_head_session_reference.py','run_deferred_zero_session_reference.py'),
    ('compare_fused_head_argmax.py','compare_deferred_state_zero.py'),
    ('run_fused_head_argmax_profiles.py','run_deferred_state_zero_profiles.py')):
    source=source.replace(old,new)
needle='libraries={';assert source.count(needle)==1
source=source.replace(needle,'''for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    p=root/'models'/model/'config.json';entries[str(p.relative_to(root))]=p
    for folder in (root/'logs'/native_trial/'native').glob(model+'-b*-attempt*'):
        metadata_path=folder/'metadata.json'
        if metadata_path.exists():assert digest(p)==json.loads(metadata_path.read_text())['assets'][model]['config.json']
'''+needle)
exec(compile(source,__file__,'exec'))
