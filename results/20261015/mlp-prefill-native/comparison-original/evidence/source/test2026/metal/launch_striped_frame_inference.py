"""Start actual inference only after both terminal CPU trials pass independent audit."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for name in ('striped-frame-probe-20261015','striped-frame-probe-confirm-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
    summary=json.loads((root/'logs'/name/'jobs/summary.json').read_text())
    for n,sha in summary['source_sha256'].items():assert hashlib.sha256((root/'source/test2026/metal'/n).read_bytes()).hexdigest()==sha
audit=json.loads((root/'work/striped-frame-cpu-verification.json').read_text())
assert len(audit['trials'])==2 and all(v['comparisons']==960 and v['real_image_samples']==192 for v in audit['trials'])
source=Path(__file__).with_name('launch_decode_matrix_probe.py').read_text().replace('run_decode_matrix_probe.py','run_striped_frame_inference.py')
exec(compile(source,__file__,'exec'))
