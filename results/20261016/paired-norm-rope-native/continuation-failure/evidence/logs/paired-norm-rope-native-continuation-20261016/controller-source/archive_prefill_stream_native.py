"""Archive complete original validation and exact streamed-prefill profiles."""
from pathlib import Path
source=Path(__file__).with_name('archive_dirty_prefix_reset_native.py').read_text()
source=source.replace('dirty-prefix-reset-checks-20261014','prefill-stream-checks-20261015').replace('dirty-prefix-reset-native-20261014','prefill-stream-native-20261015').replace('dirty-prefix-reset-staging','prefill-stream-staging').replace('dirty-prefix-reset-original-source-20261014','prefill-stream-original-source-20261014').replace('build-dirty-prefix-reset','build-prefill-stream')
start=source.index("gates=root/");end=source.index('for folder in ',start);source=source[:start]+source[end:]
needle="for name in set(manifest)|"
source=source.replace(needle,"scripts=scripts+('run_guarded_prefill_stream_checks','launch_prefill_stream_checks','compare_prefill_stream','run_prefill_stream_profiles','queue_prefill_stream_profiles','archive_prefill_stream_native')\n"+needle)
exec(compile(source,__file__,'exec'))
