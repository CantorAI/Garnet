"""Preserve the first native source before installing row-copy admission repair."""
from pathlib import Path
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','dirty-prefix-reset-repair-staging').replace('residual-norm-original-source-20261014','dirty-prefix-reset-repair-original-source-20261014')
exec(compile(source,str(Path(__file__)),'exec'))
