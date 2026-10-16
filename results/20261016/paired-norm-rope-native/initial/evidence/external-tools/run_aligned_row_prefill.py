"""Private zero-padded storage and original common-alpha MPS at aligned row counts."""
from pathlib import Path
source=Path(__file__).with_name('run_packed_body_prefill.py').read_text()
source=source.replace('make_packed_body_prefill','make_aligned_row_prefill')
source=source.replace('packed_body_prefill_gpu','aligned_row_prefill_gpu')
source=source.replace('metal_context_packed_prefill','metal_context_aligned_rows')
source=source.replace('Packed prefill','Aligned-row prefill').replace('packed prefill','aligned-row prefill')
source=source.replace('original BF16 values in four-output storage','original exact common-alpha half weight storage, input/output backing rows padded to128 while public rows stay logical')
source=source.replace('FP32 tiles','aligned MPS rows')
needle='exec(compile(source,__file__,\'exec\'))';assert source.count(needle)==1
source=source.replace(needle,"source=source.replace(\"paths=[Path(__file__),\",\"paths=[Path(__file__),s/'run_packed_body_prefill.py',\")\nsource=source.replace('scope=__doc__','scope='+repr('Private zero-backed physical MPS prefill alignment; original logical rows, learned values and FP32 arithmetic.'))\n"+needle)
exec(compile(source,__file__,'exec'))
