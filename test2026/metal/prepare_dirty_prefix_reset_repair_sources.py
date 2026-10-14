"""Pin the copy-tracing repair against the exact first native implementation."""
from pathlib import Path
source=Path(__file__).with_name('prepare_dirty_prefix_reset_sources.py').read_text().replace('dirty-prefix-reset-staging','dirty-prefix-reset-repair-staging').replace('4053c7d4','32b90ea8').replace("if name!='test2026/metal/dirty_prefix_reset_test.cpp':",'if True:').replace('install_dirty_prefix_reset_sources.py','install_dirty_prefix_reset_repair_sources.py')
exec(compile(source,str(Path(__file__)),'exec'))
