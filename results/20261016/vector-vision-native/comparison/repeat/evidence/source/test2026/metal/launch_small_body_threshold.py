"""Wait on exact hybrid graph job before any private decode work."""
from pathlib import Path

source=Path(__file__).with_name('launch_interleaved_bf16_body.py').read_text()
source=source.replace("dependencies={'wide-batch-prefill-checks-20261015':69725,'wide-batch-prefill-native-20261015':72436}","dependencies={'hybrid-combined-graph-prefill-20261015':35785}")
source=source.replace("runner='run_interleaved_bf16_body.py'","runner='run_small_body_threshold.py'")
exec(compile(source,__file__,'exec'))
