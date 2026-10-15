"""Retry a proven pre-worker log-path failure without changing the validated binary."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/prefill-stream-checks-20261015/exit.json').read_text())['exit_code']==0
assert json.loads((root/'logs/prefill-stream-native-20261015/exit.json').read_text())['exit_code']==1
failure=(root/'logs/prefill-stream-native-20261015/jobs/Qwen3-0.6B-b1-attempt0.log').read_text()
assert 'FileNotFoundError' in failure and 'prefill-stream-checks-20261014/jobs/metadata.json' in failure
assert hashlib.sha256((root/'build-prefill-stream/libgarnet.dylib').read_bytes()).hexdigest()=='bad8c5db120df0e6ac107a0fabc6b19708a379f36785b7b5c8869a30fd14534b'
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_prefill_stream_profiles.py')
exec(compile(source,__file__,'exec'))
