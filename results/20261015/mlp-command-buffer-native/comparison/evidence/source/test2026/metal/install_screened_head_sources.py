"""Require archived terminal component jobs; preserve exact existing native source."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('screened-head-guarded-20261015','screened-head-int6-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
assert not (root/'source/src/backends/metal/screened_head_shader.h').exists()
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','screened-head-staging').replace('residual-norm-original-source-20261014','screened-head-original-source-20261015').replace('residual-norm','screened-head')
exec(compile(source,__file__,'exec'))
