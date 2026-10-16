"""Freeze exact asynchronous-key graph trials, including failures."""
from pathlib import Path

wrapper=Path(__file__).with_name('archive_hybrid_combined_graph_prefill.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("['hybrid-combined-graph-prefill-20261015']","['async-key-graph-prefill-20261015']")
source=source.replace("'archive_hybrid_combined_graph_prefill.py','launch_hybrid_combined_graph_prefill.py'","'archive_async_key_graph_prefill.py','launch_async_key_graph_prefill.py'")
exec(compile(source,__file__,'exec'))
