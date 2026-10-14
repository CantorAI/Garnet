"""Preserve original source bytes and reject unexpected new paths before install."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();stage=root/'vision-global-alpha-staging'
original=json.loads((stage/'original.json').read_text());candidate=json.loads((stage/'manifest.json').read_text())
assert all(not (root/'source'/name).exists() for name in set(candidate)-set(original))
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','vision-global-alpha-staging').replace('residual-norm-original-source-20261014','vision-global-alpha-original-source-20261014').replace('residual-norm sources','vision-global-alpha sources')
exec(compile(source,str(Path(__file__)),'exec'))
