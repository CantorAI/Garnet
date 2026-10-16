"""Wait on exact live native gates/benchmark, then exclusive combined-graph probe."""
from pathlib import Path
source=Path(__file__).with_name('launch_interleaved_bf16_body.py').read_text()
needle="dependencies={'wide-batch-prefill-checks-20261015':69725,'wide-batch-prefill-native-20261015':72436}";assert source.count(needle)==1
source=source.replace(needle,"dependencies={'coalesced-body-native-checks-20261015':88930,'coalesced-body-native-20261015':88983}")
source=source.replace("runner='run_interleaved_bf16_body.py'","runner='run_combined_graph_prefill.py'")
exec(compile(source,__file__,'exec'))
