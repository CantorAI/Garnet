"""Install pinned optional candidate after all component jobs ended."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('fused-head-argmax-20261015','fused-head-argmax-confirm-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','fused-head-argmax-staging').replace('residual-norm-original-source-20261014','fused-head-argmax-original-source-20261015').replace('residual-norm','fused-head-argmax')
exec(compile(source,__file__,'exec'))
