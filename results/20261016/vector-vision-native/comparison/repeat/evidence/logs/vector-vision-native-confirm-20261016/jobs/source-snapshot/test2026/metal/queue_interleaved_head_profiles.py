"""Wait for the verified owned 27-stage candidate before any serving benchmark."""
from pathlib import Path
wrapper=Path(__file__).with_name('queue_small_seed_profiles.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('small-seed-checks-20261015','interleaved-head-checks-20261015'),('small-seed-staging/manifest.json','interleaved-head-staging/manifest.json'),('queue_small_seed_profiles.py','queue_interleaved_head_profiles.py'),('run_small_seed_profiles.py','run_interleaved_head_profiles.py'),('compare_small_seed.py','compare_interleaved_head.py'),('run_guarded_small_seed_checks.py','run_guarded_interleaved_head_checks.py'),('launch_small_seed_checks.py','launch_interleaved_head_checks.py')]:source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
