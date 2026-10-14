"""Install only after the failed attempt's source identities still match."""
from pathlib import Path
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','single-coder-head-repair-staging').replace('residual-norm-original-source-20261014','single-coder-head-repair-original-source-20261014').replace('residual-norm sources','single-coder-head repaired sources')
exec(compile(source,str(Path(__file__)),'exec'))
