"""Install only after owned component trials ended; preserve original files."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('shape-specialized-bf16-20261015','shape-specialized-bf16-confirm-20261015','reused-zero-source-confirm-native-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
assert not (root/'source/src/backends/metal/body_shape_shader.h').exists()
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','body-shape-staging').replace('residual-norm-original-source-20261014','body-shape-original-source-20261015').replace('residual-norm','body-shape')
exec(compile(source,__file__,'exec'))
