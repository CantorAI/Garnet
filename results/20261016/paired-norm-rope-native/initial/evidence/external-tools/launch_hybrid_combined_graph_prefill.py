"""Wait for exact matched confirmation before exclusive original-key graph study."""
from pathlib import Path

source=Path(__file__).with_name('launch_interleaved_bf16_body.py').read_text()
source=source.replace("dependencies={'wide-batch-prefill-checks-20261015':69725,'wide-batch-prefill-native-20261015':72436}","dependencies={'coalesced-body-vllm-confirm-20261015':34285}")
source=source.replace("runner='run_interleaved_bf16_body.py'","runner='run_hybrid_combined_graph_prefill.py'")
exec(compile(source,__file__,'exec'))
