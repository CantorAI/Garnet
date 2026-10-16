"""Wait for all 28 native body gates before sequential complete-model ABBA."""
from pathlib import Path
wrapper=Path(__file__).with_name('queue_wide_batch_prefill_profiles.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('wide-batch-prefill-checks-20261015','interleaved-body-native-checks-20261015'),('wide-batch-prefill-staging','interleaved-body-native-staging'),('queue_wide_batch_prefill_profiles.py','queue_interleaved_body_native_profiles.py'),('run_wide_batch_prefill_profiles.py','run_interleaved_body_native_profiles.py'),('compare_wide_batch_prefill.py','compare_interleaved_body_native.py'),('run_guarded_wide_batch_prefill_checks.py','run_guarded_interleaved_body_native_checks.py'),('launch_wide_batch_prefill_checks.py','launch_interleaved_body_native_checks.py')]:source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
