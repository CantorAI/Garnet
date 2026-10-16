"""Require exact prior dependent MLP job terminal before corrected reference run."""
from pathlib import Path

source=Path(__file__).with_name('launch_interleaved_bf16_body.py').read_text()
source=source.replace("dependencies={'wide-batch-prefill-checks-20261015':69725,'wide-batch-prefill-native-20261015':72436}","dependencies={'dependent-mlp-graph-20261015':52062}")
source=source.replace("runner='run_interleaved_bf16_body.py'","runner='run_stage_rounded_mlp_graph.py'")
exec(compile(source,__file__,'exec'))
