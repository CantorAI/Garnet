"""Wait for exact trained-image/original/recovery gates before full native ABBA."""
from pathlib import Path
source=Path(__file__).with_name('queue_screened_prefill_reset_profiles.py').read_text()
for old,new in [('screened-prefill-reset-checks-20261015','serial-vision-checks-20261015'),('pid=29706','pid=64980'),('len(stages)==26','len(stages)==27'),('screened-head-staging/manifest.json','serial-vision-staging/manifest.json'),('queue_screened_prefill_reset_profiles.py','queue_serial_vision_profiles.py'),('run_screened_prefill_reset_profiles.py','run_serial_vision_profiles.py'),('compare_screened_prefill_reset.py','compare_serial_vision.py'),('run_guarded_screened_prefill_reset_checks.py','run_guarded_serial_vision_checks.py'),('launch_screened_prefill_reset_checks.py','launch_serial_vision_checks.py')]:source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
