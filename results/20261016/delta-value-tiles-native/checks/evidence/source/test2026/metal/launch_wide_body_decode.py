"""Require previous native and prefill studies terminal, then exclusive decode probe."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('interleaved-body-native-confirm-20261015','mpsgraph-prefill-repaired-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text()
source=source.replace('run_guarded_global_half_alpha_checks.py','run_wide_body_decode.py')
exec(compile(source,__file__,'exec'))
