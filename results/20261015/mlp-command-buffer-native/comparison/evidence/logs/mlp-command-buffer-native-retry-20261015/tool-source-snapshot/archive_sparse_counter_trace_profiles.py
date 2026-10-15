"""Archive the sparse diagnostic with the same full source and checkpoint proof."""
from pathlib import Path
source=Path(__file__).with_name('archive_counter_trace_profiles.py').read_text()
source=source.replace('counter-trace-file-profiles-20261015','counter-trace-sparse-profiles-20261015').replace('counter-trace-file-source','counter-trace-sparse-source').replace('build-counter-trace-file','build-counter-trace-sparse')
source=source.replace("'launch_counter_trace_profiles.py','archive_counter_trace_profiles.py'", "'launch_counter_trace_profiles.py','launch_sparse_counter_trace_profiles.py','archive_counter_trace_profiles.py','archive_sparse_counter_trace_profiles.py'")
exec(compile(source,__file__,'exec'))
