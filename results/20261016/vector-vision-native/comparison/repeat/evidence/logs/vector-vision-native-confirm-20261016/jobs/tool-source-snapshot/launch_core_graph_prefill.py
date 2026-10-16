"""Require terminal exact async graph confirmation before core compilation study."""
from pathlib import Path

source=Path(__file__).with_name('launch_interleaved_bf16_body.py').read_text()
source=source.replace("dependencies={'wide-batch-prefill-checks-20261015':69725,'wide-batch-prefill-native-20261015':72436}","dependencies={'async-key-graph-prefill-confirm-20261015':48881}")
source=source.replace("runner='run_interleaved_bf16_body.py'","runner='run_core_graph_prefill.py'")
exec(compile(source,__file__,'exec'))
