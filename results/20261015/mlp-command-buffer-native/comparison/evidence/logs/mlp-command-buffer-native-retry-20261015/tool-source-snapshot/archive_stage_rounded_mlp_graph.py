"""Keep stage-rounded MLP evidence separate from whole-FP64 stress evidence."""
from pathlib import Path

wrapper=Path(__file__).with_name('archive_dependent_mlp_graph.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("['dependent-mlp-graph-20261015']","['stage-rounded-mlp-graph-20261015']")
source=source.replace("'archive_dependent_mlp_graph.py','launch_dependent_mlp_graph.py'","'archive_stage_rounded_mlp_graph.py','launch_stage_rounded_mlp_graph.py'")
exec(compile(source,__file__,'exec'))
