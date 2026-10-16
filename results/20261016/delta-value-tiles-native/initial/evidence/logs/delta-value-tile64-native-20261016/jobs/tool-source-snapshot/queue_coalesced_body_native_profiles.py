"""Require the exact28 successful native stages before all six serving profiles."""
from pathlib import Path
wrapper=Path(__file__).with_name('queue_interleaved_body_native_profiles.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('interleaved-body-native-checks-20261015','coalesced-body-native-checks-20261015'),('interleaved-body-native-staging','coalesced-body-native-staging'),('queue_interleaved_body_native_profiles.py','queue_coalesced_body_native_profiles.py'),('run_interleaved_body_native_profiles.py','run_coalesced_body_native_profiles.py'),('compare_interleaved_body_native.py','compare_coalesced_body_native.py'),('run_guarded_interleaved_body_native_checks.py','run_guarded_coalesced_body_native_checks.py'),('launch_interleaved_body_native_checks.py','launch_coalesced_body_native_checks.py')]:source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
