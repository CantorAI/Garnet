"""Preserve terminal combined graph trials, including failures and original assets."""
from pathlib import Path
source=Path(__file__).with_name('archive_mpsgraph_prefill.py').read_text()
source=source.replace("['mpsgraph-prefill-20261015']","['combined-graph-prefill-20261015']")
source=source.replace('build-interleaved-bf16-body','build-coalesced-bf16-body')
source=source.replace("'archive_mpsgraph_prefill.py','launch_mpsgraph_prefill.py'","'archive_combined_graph_prefill.py','launch_combined_graph_prefill.py'")
exec(compile(source,__file__,'exec'))
