"""Bounded live-handle/build/CTest/source proof; never mark pending work complete."""
from pathlib import Path
source=Path(__file__).with_name('snapshot_dirty_prefix_reset_native.py').read_text()
source=source.replace('dirty-prefix-reset-checks-20261014','prefill-stream-checks-20261015').replace('dirty-prefix-reset-native-20261014','prefill-stream-native-20261015').replace('launch_dirty_prefix_reset_checks.py','launch_prefill_stream_checks.py').replace('queue_dirty_prefix_reset_profiles.py','queue_prefill_stream_profiles.py').replace('44364','98669').replace('45551','3161').replace('dirty-prefix-reset-staging','prefill-stream-staging').replace('build-dirty-prefix-reset','build-prefill-stream')
source=source.replace('Dirty prefix reset complete CPU/GPU positive zero, bounds, uninitialized and abort PASS 216','Prefill stream exact dependencies, ownership, abort drain and recovery PASS 27')
exec(compile(source,__file__,'exec'))
