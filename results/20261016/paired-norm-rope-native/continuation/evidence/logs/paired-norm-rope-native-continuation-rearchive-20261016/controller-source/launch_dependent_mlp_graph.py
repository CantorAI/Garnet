"""Require terminal exact core graph job before exclusive dependent MLP study."""
from pathlib import Path

source=Path(__file__).with_name('launch_interleaved_bf16_body.py').read_text()
source=source.replace("dependencies={'wide-batch-prefill-checks-20261015':69725,'wide-batch-prefill-native-20261015':72436}","dependencies={'core-graph-prefill-20261015':50246}")
source=source.replace("runner='run_interleaved_bf16_body.py'","runner='run_dependent_mlp_graph.py'")
exec(compile(source,__file__,'exec'))
