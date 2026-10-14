"""Prepare source changes against the completed row-specialization branch base."""
from pathlib import Path
source=Path(__file__).with_name('prepare_packed_prefill_norm_sources.py').read_text()
# Expand the small preparation wrapper before substituting its branch base.
base=Path(__file__).with_name('prepare_residual_norm_sources.py').read_text()
base=base.replace('residual-norm-staging','global-half-alpha-staging').replace('8157a4d5:','794fe78a:')
base=base.replace("'test2026/metal/matmul_test.cpp']","'test2026/metal/matmul_test.cpp','xModel/qwen/metal_dense/decoder.py','xModel/qwen35/vl_0_8b/metal_decoder.py']")
exec(compile(base,str(Path(__file__)),'exec'))
