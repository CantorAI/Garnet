"""Install after joint component and route-policy inference jobs are terminal."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('joint-state-zero-20261015','joint-state-zero-confirm-20261015','decode-only-head-confirm-native-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','deferred-state-zero-staging').replace('residual-norm-original-source-20261014','deferred-state-zero-original-source-20261015').replace('residual-norm','deferred-state-zero')
exec(compile(source,__file__,'exec'))
