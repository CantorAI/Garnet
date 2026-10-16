"""Original gates and vision bits plus actual small-seed trained recovery checks."""
from pathlib import Path
wrapper=Path(__file__).with_name('run_guarded_serial_vision_checks.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('build-serial-vision-prefetch','build-small-screen-seed')
source=source.replace("'METAL_SERIAL_VISION_PREFETCH'","'METAL_SERIAL_VISION_PREFETCH','METAL_SMALL_SCREEN_SEED'")
source=source.replace('test_screened_head_session_reference.py','test_small_seed_session_reference.py')
exec(compile(source,__file__,'exec'))
