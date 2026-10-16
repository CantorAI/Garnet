"""Repeat both runtime/frontend profiles after independent first-run output audit."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/striped-frame-inference-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/striped-frame-inference-verification.json').read_text())
assert audit['actual_fp32_gpu_audits']==1 and audit['complete_matched_requests']==48 and len(audit['profiles'])==2
source=Path(__file__).with_name('launch_striped_frame_inference.py').read_text()
exec(compile(source,__file__,'exec'))
