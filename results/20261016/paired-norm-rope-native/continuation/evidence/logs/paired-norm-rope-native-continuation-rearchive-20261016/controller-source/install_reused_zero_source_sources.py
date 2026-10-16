"""Install after all component and full deferred-reset runs ended."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('deferred-state-zero-confirm-native-20261015','reused-zero-source-20261015','reused-zero-source-confirm-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','reused-zero-source-staging').replace('residual-norm-original-source-20261014','reused-zero-source-original-source-20261015').replace('residual-norm','reused-zero-source')
exec(compile(source,__file__,'exec'))
