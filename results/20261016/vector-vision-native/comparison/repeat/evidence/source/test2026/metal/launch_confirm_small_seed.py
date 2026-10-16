"""Repeat all six native profiles after the first independent small-seed audit."""
from pathlib import Path
wrapper=Path(__file__).with_name('launch_confirm_serial_vision.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('serial-vision-native-20261015','small-seed-native-20261015'),('serial-vision-native-verification.json','small-seed-native-verification.json'),('serial-vision-checks-20261015','small-seed-checks-20261015'),('build-serial-vision-prefetch','build-small-screen-seed'),('serial-vision-staging/manifest.json','small-seed-staging/manifest.json'),('run_serial_vision_profiles.py','run_small_seed_profiles.py')]:source=source.replace(old,new)
source=source.replace("assert audit['trained_full_grid_profiles']==4","assert audit['actual_candidate_seed_rows']==1024 and audit['trained_full_grid_profiles']==4")
exec(compile(source,__file__,'exec'))
