"""Wait for the exact repeated native job before an exclusive aligned-row probe."""
from pathlib import Path
import json
source=Path(__file__).with_name('launch_interleaved_bf16_body.py').read_text()
needle="dependencies={'wide-batch-prefill-checks-20261015':69725,'wide-batch-prefill-native-20261015':72436}";assert source.count(needle)==1
source=source.replace(needle,"dependencies={'interleaved-body-native-confirm-20261015':79571}")
source=source.replace("runner='run_interleaved_bf16_body.py'","runner='run_aligned_row_prefill.py'")
exec(compile(source,__file__,'exec'))
