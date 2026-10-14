"""Preserve exact prior source before installing optional partial submissions."""
from pathlib import Path
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','prefill-stream-staging').replace('residual-norm-original-source-20261014','prefill-stream-original-source-20261014')
exec(compile(source,__file__,'exec'))
