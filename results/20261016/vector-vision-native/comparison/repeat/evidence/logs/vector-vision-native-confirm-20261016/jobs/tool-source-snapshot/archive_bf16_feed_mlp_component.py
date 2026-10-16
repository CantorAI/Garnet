"""Preserve trained BF16-feed graph outcomes and original learned asset proof."""
from pathlib import Path
p=Path(__file__).with_name('archive_stage_rounded_mlp_graph.py');wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns);source=ns['source']
source=source.replace("['stage-rounded-mlp-graph-20261015']","['bf16-feed-mlp-component-20261015']")
source=source.replace("'archive_stage_rounded_mlp_graph.py','launch_stage_rounded_mlp_graph.py'","'archive_bf16_feed_mlp_component.py','launch_bf16_feed_mlp_component.py'")
exec(compile(source,__file__,'exec'))
