"""Repeat the frozen async-key component after independent numerical/timing audit."""
from pathlib import Path
import hashlib,json,sys

root=Path(sys.argv[1]).resolve();prior=root/'logs/async-key-graph-prefill-20261015'
assert json.loads((prior/'pid.json').read_text())['pid']==47481
assert json.loads((prior/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/async-key-graph-prefill-verification.json').read_text())
assert audit['numeric_checks']==48 and audit['first_graph_runs']==12 and audit['timed_executions']==768
meta=json.loads((prior/'jobs/metadata.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in meta['source_sha256'].items())
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_async_key_graph_prefill.py')
exec(compile(source,__file__,'exec'))
