"""Bounded proof of repaired build/CTest/copy gates and exact live dependency."""
from pathlib import Path
source=Path(__file__).with_name('snapshot_dirty_prefix_reset_native.py').read_text()
source=source.replace('dirty-prefix-reset-checks-20261014','dirty-prefix-reset-repair-checks-20261014').replace('dirty-prefix-reset-native-20261014','dirty-prefix-reset-repair-native-20261014').replace('launch_dirty_prefix_reset_checks.py','launch_dirty_prefix_reset_repair_checks.py').replace('queue_dirty_prefix_reset_profiles.py','queue_dirty_prefix_reset_repair_profiles.py').replace('44364','66790').replace('45551','68433').replace('dirty-prefix-reset-staging','dirty-prefix-reset-repair-staging').replace('build-dirty-prefix-reset','build-dirty-prefix-reset-repaired')
exec(compile(source,__file__,'exec'))
