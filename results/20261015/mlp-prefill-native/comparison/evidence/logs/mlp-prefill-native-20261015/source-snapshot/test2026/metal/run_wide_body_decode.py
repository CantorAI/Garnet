"""Eight-output BF16 decode layout, original B1/B2 and exact native output bits."""
from pathlib import Path
source=Path(__file__).with_name('run_packed_body_prefill.py').read_text()
for old,new in [('make_packed_body_prefill','make_wide_body_decode'),('packed_body_prefill_gpu','wide_body_decode_gpu'),('metal_context_packed_prefill','metal_context_wide_body')]:source=source.replace(old,new)
source=source.replace("rows=80*batch if model=='Qwen3.5-0.8B' else (24 if batch==1 else 44) if model=='Qwen3-0.6B' else 26*batch","rows=batch")
source=source.replace('packed prefill','wide body decode')
needle='exec(compile(source,__file__,\'exec\'))';assert source.count(needle)==1
source=source.replace(needle,"source=source.replace(\"paths=[Path(__file__),\",\"paths=[Path(__file__),s/'run_packed_body_prefill.py',s/'wide_bf16_body_shader.h',\")\nsource=source.replace('scope=__doc__','scope='+repr('Private eight-output BF16 decode versus current four-output layout, original B1/B2 and exact output bits.'))\nsource=source.replace('original BF16 values in four-output storage','original BF16 values in four/eight-output storage;64/128/256-thread variants')\nsource=source.replace('Frozen physical prefill rows24/44,26/52,80/160','Original decode rows1/2')\n"+needle)
exec(compile(source,__file__,'exec'))
