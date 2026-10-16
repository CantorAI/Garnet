"""Require the audited BF16-storage primitive before trained timing admission."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
audit=json.loads((root/'mlp-bf16-feed-primitive-verification.json').read_text())
assert audit['original_primitive_cases']==64 and audit['dependent_two_graph_cases']==64
assert json.loads((root/'logs/mlp-bf16-feed-primitive-20261015/exit.json').read_text())['exit_code']==0
with (root/'mlp-bf16-feed-primitive-20261015.tar.gz').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()==audit['archive_sha256']
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('component',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_bf16_feed_mlp_component.py'")
exec(compile(source,__file__,'exec'))
