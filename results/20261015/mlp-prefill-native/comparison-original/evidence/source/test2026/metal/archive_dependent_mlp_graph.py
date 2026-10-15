"""Preserve dependent MLP graph trials and original learned tensor identities."""
from pathlib import Path

wrapper=Path(__file__).with_name('archive_combined_graph_prefill.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("['combined-graph-prefill-20261015']","['dependent-mlp-graph-20261015']")
source=source.replace("'archive_combined_graph_prefill.py','launch_combined_graph_prefill.py'","'archive_dependent_mlp_graph.py','launch_dependent_mlp_graph.py'")
exec(compile(source,__file__,'exec'))
