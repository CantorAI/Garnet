"""Preserve terminal small-seed native gates, original sources and complete ABBA."""
from pathlib import Path
wrapper=Path(__file__).with_name('archive_serial_vision.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('serial-vision-checks-20261015','small-seed-checks-20261015'),('serial-vision-native-20261015','small-seed-native-20261015'),('build-serial-vision-prefetch','build-small-screen-seed'),('build-screened-prefill-reset','build-serial-vision-prefetch'),('serial-vision-staging','small-seed-staging'),('serial-vision-original-source-20261015','small-seed-original-source-20261015'),('75b58100ea5db4f3f1a4ee4ef18a1d3c4c0d02288a779a6bacf38ce5f91dad0e','b61c4732d667ebb0c813d8d67e0b132ace611940700d211b65fb6871bcb79e0d')]:source=source.replace(old,new)
for old,new in [('archive_serial_vision.py','archive_small_seed.py'),('run_guarded_serial_vision_checks.py','run_guarded_small_seed_checks.py'),('launch_serial_vision_checks.py','launch_small_seed_checks.py'),('compare_serial_vision.py','compare_small_seed.py'),('run_serial_vision_profiles.py','run_small_seed_profiles.py')]:source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
