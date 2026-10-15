"""Fresh six-profile repeat after original gates and first native audit pass."""
from pathlib import Path
source=Path(__file__).with_name('launch_confirm_screened_prefill_reset.py').read_text()
for old,new in [('screened-prefill-reset-native-20261015','serial-vision-native-20261015'),('screened-prefill-reset-verification.json','serial-vision-native-verification.json'),('screened-prefill-reset-checks-20261015','serial-vision-checks-20261015'),('build-screened-prefill-reset','build-serial-vision-prefetch'),('screened-head-staging/manifest.json','serial-vision-staging/manifest.json'),('run_screened_prefill_reset_profiles.py','run_serial_vision_profiles.py')]:source=source.replace(old,new)
source=source.replace("assert audit['original_validation_stages']==19","assert audit['trained_full_grid_profiles']==4 and audit['original_validation_stages']==19")
exec(compile(source,__file__,'exec'))
