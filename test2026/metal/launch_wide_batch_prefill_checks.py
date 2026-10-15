"""Launch exclusively after preceding native jobs are terminal and archived."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('interleaved-head-native-20261015','interleaved-head-native-confirm-20261015','replanned-vl-prefill-20261015','replanned-vl-prefill-confirm-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0,name
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_guarded_wide_batch_prefill_checks.py')
exec(compile(source,__file__,'exec'))
