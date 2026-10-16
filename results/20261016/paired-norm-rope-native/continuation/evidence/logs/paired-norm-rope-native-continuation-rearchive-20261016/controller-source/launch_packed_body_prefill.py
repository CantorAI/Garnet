"""Wait on the exact native validation/serving handles before an isolated prefill probe."""
from pathlib import Path
source=Path(__file__).with_name('launch_interleaved_bf16_body.py').read_text()
needle="dependencies={'wide-batch-prefill-checks-20261015':69725,'wide-batch-prefill-native-20261015':72436}";assert source.count(needle)==1
source=source.replace(needle,"dependencies={'interleaved-body-native-checks-20261015':49634,'interleaved-body-native-20261015':49699}")
source=source.replace("runner='run_interleaved_bf16_body.py'","runner='run_packed_body_prefill.py'")
exec(compile(source,__file__,'exec'))
