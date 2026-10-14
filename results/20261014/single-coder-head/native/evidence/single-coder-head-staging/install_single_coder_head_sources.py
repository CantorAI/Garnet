"""Install pinned head experiment after preserving every original source byte."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();stage=root/'single-coder-head-staging'
original=json.loads((stage/'original.json').read_text());candidate=json.loads((stage/'manifest.json').read_text())
assert all(not (root/'source'/name).exists() for name in set(candidate)-set(original))
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','single-coder-head-staging').replace('residual-norm-original-source-20261014','single-coder-head-original-source-20261014').replace('residual-norm sources','single-coder-head sources')
exec(compile(source,str(Path(__file__)),'exec'))
