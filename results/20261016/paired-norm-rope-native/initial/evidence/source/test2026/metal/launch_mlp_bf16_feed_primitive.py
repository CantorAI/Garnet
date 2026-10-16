"""Admit the new storage probe only after selected-shape native work is archived."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();alt=root/'shape-selected-two-investigation-20261015'
trial=alt/'logs/shape-selected-two-native-20261015'
assert json.loads((trial/'exit.json').read_text())['exit_code']==0
archive=alt/'shape-selected-two-native-20261015.tar.gz'
audit=json.loads((alt/'shape-selected-two-native-verification.json').read_text())
assert audit['full_output_requests']==72 and audit['actual_selected_two_routes']
with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==audit['archive_sha256']
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('primitive',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_mlp_bf16_feed_primitive.py'")
exec(compile(source,__file__,'exec'))
