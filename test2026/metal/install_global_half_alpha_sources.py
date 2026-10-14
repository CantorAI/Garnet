"""Install only pinned global-alpha changes, retaining exact original bytes."""
from pathlib import Path
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text()
source=source.replace('residual-norm-staging','global-half-alpha-staging').replace('residual-norm-original-source-20261014','global-half-alpha-original-source-20261014')
source=source.replace('residual-norm sources','global-half-alpha sources')
exec(compile(source,str(Path(__file__)),'exec'))
