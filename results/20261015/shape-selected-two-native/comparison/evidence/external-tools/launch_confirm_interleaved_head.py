"""Repeat all six profiles after terminal native work and independent audit."""
from pathlib import Path
wrapper=Path(__file__).with_name('launch_confirm_small_seed.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('small-seed-native-20261015','interleaved-head-native-20261015'),('small-seed-native-verification.json','interleaved-head-native-verification.json'),('small-seed-checks-20261015','interleaved-head-checks-20261015'),('build-small-screen-seed','build-interleaved-screen-index'),('small-seed-staging/manifest.json','interleaved-head-staging/manifest.json'),('run_small_seed_profiles.py','run_interleaved_head_profiles.py')]:source=source.replace(old,new)
source=source.replace("assert audit['actual_candidate_seed_rows']==1024","assert audit['actual_interleaved'] and audit['actual_candidate_seed_rows']==1024")
exec(compile(source,__file__,'exec'))
