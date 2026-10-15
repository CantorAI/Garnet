"""Original 27 gates plus actual layout counts and complete trained index recovery."""
from pathlib import Path
wrapper=Path(__file__).with_name('run_guarded_small_seed_checks.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('build-small-screen-seed','build-interleaved-screen-index')
source=source.replace("'METAL_SMALL_SCREEN_SEED'","'METAL_SMALL_SCREEN_SEED','METAL_INTERLEAVED_SCREEN_INDEX'")
source=source.replace('test_small_seed_session_reference.py','test_interleaved_head_session_reference.py')
exec(compile(source,__file__,'exec'))
