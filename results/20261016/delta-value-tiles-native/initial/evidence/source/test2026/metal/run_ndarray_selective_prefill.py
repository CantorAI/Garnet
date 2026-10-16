"""Selective no-exact-half NDArray at all six original trained prefill shapes."""
from pathlib import Path
import ast, sys
p=Path(__file__).with_name('run_ndarray_prefill.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.index('destination=Path(sys.argv[2])')],str(p),'exec'),ns)
source=ns['source']
old='from make_ndarray_prefill import context as generate_context,header as generate_header,harness as generate_harness'
assert source.count(old)==1
source=source.replace(old,old.replace('make_ndarray_prefill','make_ndarray_selective_prefill'))
source=source.replace("s/'make_ndarray_prefill.py',", "s/'make_ndarray_selective_prefill.py',s/'make_ndarray_prefill.py',s/'run_ndarray_prefill.py',")
source=source.replace("scope='Streamed original trained projection sets in Garnet MetalContext. Policy1 exact-half alias with three-source alpha; policy2 exact expanded FP32 cache with two-source unit alpha; policy3 native null.",
    "scope='Streamed original trained projection sets in Garnet MetalContext. Policy1 full expanded F32 NDArray; policy2 only matrices lacking common AND row-wise exact half; policy3 native null. Both NDArray policies use two-source unit alpha.")
source=source.replace('Mixed three-source alpha/zero addend or exact F32 weight expansion', 'Exact F32 weight expansion only')
source=source.replace('mixed policy aliases existing half storage', 'selective policy retains every existing exact-half route')
source=source.replace("ndarray_prefill_gpu", "ndarray_selective_prefill_gpu").replace('metal_context_ndarray_prefill','metal_context_ndarray_selective_prefill')
ast.parse(source)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,__file__,'exec'))
