"""Wait on the verified owned 27-stage build before starting sequential inference."""
from pathlib import Path
source=Path(__file__).with_name('queue_serial_vision_profiles.py').read_text();ns={'__file__':__file__}
exec(compile(source[:source.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('serial-vision-checks-20261015','small-seed-checks-20261015'),('pid=64980',"pid=json.loads((dependency/'pid.json').read_text())['pid']"),('serial-vision-staging/manifest.json','small-seed-staging/manifest.json'),('queue_serial_vision_profiles.py','queue_small_seed_profiles.py'),('run_serial_vision_profiles.py','run_small_seed_profiles.py'),('compare_serial_vision.py','compare_small_seed.py'),('run_guarded_serial_vision_checks.py','run_guarded_small_seed_checks.py'),('launch_serial_vision_checks.py','launch_small_seed_checks.py')]:source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
