"""Preserve terminal serial-vision gates and six-profile native ABBA evidence."""
from pathlib import Path
wrapper=Path(__file__).with_name('archive_screened_prefill_reset.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('screened-prefill-reset-checks-20261015','serial-vision-checks-20261015'),('screened-prefill-reset-native-20261015','serial-vision-native-20261015'),('build-screened-prefill-reset','build-serial-vision-prefetch'),('build-screened-head','build-screened-prefill-reset'),('screened-head-staging','serial-vision-staging'),('screened-head-original-source-20261015','serial-vision-original-source-20261015'),('2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23','75b58100ea5db4f3f1a4ee4ef18a1d3c4c0d02288a779a6bacf38ce5f91dad0e')]:source=source.replace(old,new)
for old,new in [('archive_screened_prefill_reset.py','archive_serial_vision.py'),('run_guarded_screened_prefill_reset_checks.py','run_guarded_serial_vision_checks.py'),('launch_screened_prefill_reset_checks.py','launch_serial_vision_checks.py'),('compare_screened_prefill_reset.py','compare_serial_vision.py'),('run_screened_prefill_reset_profiles.py','run_serial_vision_profiles.py')]:source=source.replace(old,new)
needle='proof=dict(files={}';assert source.count(needle)==1
source=source.replace(needle,"for fixture in (root/'logs/serial-vision-checks-20261015/jobs/full-grid').glob('*-fixture.json'):\n    p=Path(json.loads(fixture.read_text())['patch_file']);assert p.resolve().is_relative_to(root)\n    omitted[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=digest(p))\n"+needle)
exec(compile(source,__file__,'exec'))
