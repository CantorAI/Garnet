"""Same six complete-output ABBA diagnostics with reduced sampling overhead."""
from pathlib import Path
source=Path(__file__).with_name('run_counter_trace_profiles.py').read_text()
source=source.replace('build-counter-trace-file','build-counter-trace-sparse').replace('counter-trace-file-source','counter-trace-sparse-source')
source=source.replace("str(s/'make_counter_trace_source.py')","str(s/'make_sparse_counter_trace_source.py')")
source=source.replace("('run_counter_trace_profiles.py','make_counter_trace_source.py','counter_trace_helpers.inc'", "('run_sparse_counter_trace_profiles.py','run_counter_trace_profiles.py','make_sparse_counter_trace_source.py','make_counter_trace_source.py','counter_trace_helpers.inc'")
exec(compile(source,__file__,'exec'))
