"""Archive original gates, new copy tests, recovery and complete repaired profiles."""
from pathlib import Path
source=Path(__file__).with_name('archive_dirty_prefix_reset_native.py').read_text()
source=source.replace('dirty-prefix-reset-checks-20261014','dirty-prefix-reset-repair-checks-20261014').replace('dirty-prefix-reset-native-20261014','dirty-prefix-reset-repair-native-20261014').replace('dirty-prefix-reset-staging','dirty-prefix-reset-repair-staging').replace('dirty-prefix-reset-original-source-20261014','dirty-prefix-reset-repair-original-source-20261014').replace('build-dirty-prefix-reset','build-dirty-prefix-reset-repaired')
source=source.replace("'archive_dirty_prefix_reset_native')","'archive_dirty_prefix_reset_native','archive_dirty_prefix_reset_repair_native','run_guarded_dirty_prefix_reset_repair_checks','launch_dirty_prefix_reset_repair_checks','compare_dirty_prefix_reset_repair','run_dirty_prefix_reset_repair_profiles','queue_dirty_prefix_reset_repair_profiles','run_guarded_prefix_reset_repair_session_tests','run_prefix_reset_prefill_reference','test_prefix_reset_prefill_reference')")
source=source.replace("for p in gates.rglob('*.f32'):add(p)","for p in gates.rglob('*.f32'):add(p)\nfor p in (root/'logs/dirty-prefix-reset-repair-checks-20261014/jobs').rglob('*.f32'):add(p)")
exec(compile(source,__file__,'exec'))
