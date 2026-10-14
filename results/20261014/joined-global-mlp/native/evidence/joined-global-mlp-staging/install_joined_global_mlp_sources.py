"""Preserve exact prior source and reject unexpected files before installation."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();stage=root/'joined-global-mlp-staging'
original=json.loads((stage/'original.json').read_text());candidate=json.loads((stage/'manifest.json').read_text())
assert all(not (root/'source'/name).exists() for name in set(candidate)-set(original))
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text()
source=source.replace('residual-norm-staging','joined-global-mlp-staging').replace('residual-norm-original-source-20261014','joined-global-mlp-original-source-20261014')
exec(compile(source,str(Path(__file__)),'exec'))
