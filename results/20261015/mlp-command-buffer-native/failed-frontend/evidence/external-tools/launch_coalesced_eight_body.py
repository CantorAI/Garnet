"""Require archived/audited complete native comparison, then quiet decode probe."""
from pathlib import Path
import hashlib,json,sys

root=Path(sys.argv[1]).resolve()
audit=json.loads((root/'mlp-prefill-native-verification.json').read_text())
assert audit['full_output_requests']==72 and audit['actual_graph_routes']
assert audit['checks']['validation_stages']==28 and audit['checks']['ctests']==8
assert json.loads((root/'logs/mlp-prefill-native-20261015/exit.json').read_text())['exit_code']==0
with (root/'mlp-prefill-native-20261015.tar.gz').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()==audit['archive_sha256']
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('component',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_coalesced_eight_body.py'")
exec(compile(source,__file__,'exec'))
