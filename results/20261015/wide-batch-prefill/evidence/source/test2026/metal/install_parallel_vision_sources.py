"""Require terminal trials before installing hash-pinned candidate, preserving originals."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('parallel-vision-context-20261015','parallel-vision-context-confirm-20261015','threaded-striped-inference-confirm-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
assert not (root/'source/src/backends/metal/parallel_vision_context_shader.h').exists()
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','parallel-vision-staging').replace('residual-norm-original-source-20261014','parallel-vision-original-source-20261015').replace('residual-norm','parallel-vision')
exec(compile(source,__file__,'exec'))
