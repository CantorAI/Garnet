"""Wait for the repeated full native benchmark before repeating the body component."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();prior=root/'logs/interleaved-bf16-body-20261015'
assert json.loads((prior/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/interleaved-bf16-body-verification.json').read_text());assert audit['numeric_checks']==72 and audit['control_profiles']==12
meta=json.loads((prior/'jobs/metadata.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in meta['source_sha256'].items())
source=Path(__file__).with_name('launch_interleaved_bf16_body.py').read_text()
needle="dependencies={'wide-batch-prefill-checks-20261015':69725,'wide-batch-prefill-native-20261015':72436}";assert source.count(needle)==1
source=source.replace(needle,"dependencies={'wide-batch-prefill-native-confirm-20261015':json.loads((root/'logs/wide-batch-prefill-native-confirm-20261015/pid.json').read_text())['pid']}")
exec(compile(source,__file__,'exec'))
