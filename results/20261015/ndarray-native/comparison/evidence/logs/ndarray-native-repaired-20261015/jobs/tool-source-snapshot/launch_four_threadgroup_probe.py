"""Quiet geometry probe after terminal archived BF16 component work."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/bf16-feed-mlp-component-20261015/exit.json').read_text())['exit_code']==0
with (root/'bf16-feed-mlp-component-20261015.tar.gz').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='bec11fdac713dd8149993c0912b540a6f59c57008d77b46dc6ad81c1288bf361'
audit=json.loads((root/'bf16-feed-mlp-component-verification.json').read_text())
assert audit['numeric_checks']==48 and audit['timed_executions']==384
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('component',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_four_threadgroup_probe.py'")
exec(compile(source,__file__,'exec'))
