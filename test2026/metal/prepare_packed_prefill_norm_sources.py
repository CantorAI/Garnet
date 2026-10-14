"""Prepare twelve source changes against the verified residual-norm branch base."""
from pathlib import Path
source=Path(__file__).with_name('prepare_residual_norm_sources.py').read_text()
source=source.replace('residual-norm-staging','packed-prefill-norm-staging').replace('8157a4d5:','7aa695db:')
source=source.replace("'test2026/metal/matmul_test.cpp']","'test2026/metal/matmul_test.cpp','xModel/qwen/metal_dense/decoder.py','xModel/qwen35/vl_0_8b/metal_decoder.py']")
exec(compile(source,str(Path(__file__)),'exec'))
