"""Install only hash-verified packed-prefill changes, preserving prior bytes."""
from pathlib import Path
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text()
source=source.replace('residual-norm-staging','packed-prefill-norm-staging').replace('residual-norm-original-source-20261014','packed-prefill-norm-original-source-20261014')
source=source.replace('residual-norm sources','packed-prefill-norm sources')
exec(compile(source,str(Path(__file__)),'exec'))
