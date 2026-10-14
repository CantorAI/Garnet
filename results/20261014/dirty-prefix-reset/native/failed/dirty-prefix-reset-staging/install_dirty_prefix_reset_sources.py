"""Preserve exact prior source bytes before installing a pinned optional candidate."""
from pathlib import Path
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','dirty-prefix-reset-staging').replace('residual-norm-original-source-20261014','dirty-prefix-reset-original-source-20261014')
exec(compile(source,str(Path(__file__)),'exec'))
