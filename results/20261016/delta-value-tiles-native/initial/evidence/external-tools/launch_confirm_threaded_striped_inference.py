"""Repeat only after initial full-output audit and terminal GPU component work."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('threaded-striped-inference-20261015','fused-projection-groups-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/threaded-striped-inference-verification.json').read_text())
assert audit['actual_fp32_gpu_audits']==1 and audit['complete_matched_requests']==48 and len(audit['profiles'])==2
source=Path(__file__).with_name('launch_threaded_striped_inference.py').read_text()
exec(compile(source,__file__,'exec'))
