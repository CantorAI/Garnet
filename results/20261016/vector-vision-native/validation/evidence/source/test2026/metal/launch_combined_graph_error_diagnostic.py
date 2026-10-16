"""Wait for the exact live comparator before exclusive numerical diagnosis."""
from pathlib import Path

source=Path(__file__).with_name('launch_interleaved_bf16_body.py').read_text()
source=source.replace("dependencies={'wide-batch-prefill-checks-20261015':69725,'wide-batch-prefill-native-20261015':72436}","dependencies={'coalesced-body-vllm-20261015':23200}")
source=source.replace("runner='run_interleaved_bf16_body.py'","runner='run_combined_graph_error_diagnostic.py'")
exec(compile(source,__file__,'exec'))
