"""Fresh ABBA profiles for the cache-copy admission repair."""
from pathlib import Path
ns={'__file__':__file__};wrapper=Path(__file__).with_name('compare_dirty_prefix_reset.py').read_text()
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns)
source=ns['source'].replace('build-dirty-prefix-reset','build-dirty-prefix-reset-repaired').replace('dirty-prefix-reset-checks-20261014','dirty-prefix-reset-repair-checks-20261014')
exec(compile(source,__file__,'exec'))
