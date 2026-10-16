"""Preserve terminal native layout gates, original sources and full-output ABBA."""
from pathlib import Path
wrapper=Path(__file__).with_name('archive_small_seed.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('small-seed-checks-20261015','interleaved-head-checks-20261015'),('small-seed-native-20261015','interleaved-head-native-20261015'),('build-small-screen-seed','build-interleaved-screen-index'),('build-serial-vision-prefetch','build-small-screen-seed'),('small-seed-staging','interleaved-head-staging'),('small-seed-original-source-20261015','interleaved-head-original-source-20261015'),('b61c4732d667ebb0c813d8d67e0b132ace611940700d211b65fb6871bcb79e0d','d27e746573098a1668c06a781d53396b65c6ffca87e1f4cf19e86cc5d326c876')]:source=source.replace(old,new)
for old,new in [('archive_small_seed.py','archive_interleaved_head.py'),('run_guarded_small_seed_checks.py','run_guarded_interleaved_head_checks.py'),('launch_small_seed_checks.py','launch_interleaved_head_checks.py'),('compare_small_seed.py','compare_interleaved_head.py'),('run_small_seed_profiles.py','run_interleaved_head_profiles.py')]:source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
