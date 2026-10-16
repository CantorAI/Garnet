"""Require terminal/archived/audited command-buffer cohort before the GPU probe."""
from pathlib import Path
import json,sys,tarfile

root=Path(sys.argv[1]).resolve();alt=root/'mlp-command-buffer-investigation-20261015'
controller=alt/'logs/command-buffer-native-finalize-20261015'
assert json.loads((controller/'pid.json').read_text())['pid']==13152
status=json.loads((controller/'exit.json').read_text())
assert status['exit_code']==0 and status['state']=='native_benchmark_dispatched'
dispatch=json.loads((controller/'benchmark-dispatch.json').read_text())
assert dispatch['pid']==31600
native=alt/'logs/mlp-command-buffer-native-retry-20261015'
assert json.loads((native/'pid.json').read_text())['pid']==34111
assert json.loads((native/'exit.json').read_text())['exit_code']==0
audit=json.loads((alt/'command-buffer-native-verification.json').read_text())
assert audit['full_output_requests']==72 and audit['actual_graph_routes']
assert audit['checks']['validation_stages']==28 and audit['checks']['dependent_graph_chain_cases']==64
with tarfile.open(alt/'command-buffer-native-retry-20261015.tar.gz') as tar:
    proof=json.load(tar.extractfile('evidence/manifest.json'))
assert proof['jobs']['mlp-command-buffer-native-retry-20261015']['exit_code']==0
assert proof['library_sha256']['build-mlp-command-buffer']==audit['checks']['candidate_library_sha256']
assert proof['checks_archive_sha256']==dispatch['checks_archive_sha256']
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('component',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_eight_fma_body.py'")
exec(compile(source,__file__,'exec'))
