"""Exact compressed four-output BF16 body blocks against current native layout."""
from pathlib import Path
source=Path(__file__).with_name('run_wide_body_decode.py').read_text()
for old,new in [('make_wide_body_decode','make_coalesced_body_block'),('wide_body_decode_gpu','coalesced_body_block_gpu'),('metal_context_wide_body','metal_context_coalesced_block'),('wide_bf16_body_shader.h','coalesced_body_block_shader.h')]:source=source.replace(old,new)
source=source.replace('Eight-output BF16 decode layout','Coalesced exact BF16 block decode')
source=source.replace('Private eight-output BF16 decode versus current four-output layout','Private coalesced exact12-bit BF16 body blocks versus current four-output raw layout')
source=source.replace('original BF16 values in four/eight-output storage;64/128/256-thread variants','original BF16 values in fixed-pitch sign/mantissa and exponent planes, uniform per-block raw fallback;64/128/256-thread variants')
source=source.replace("s/'run_packed_body_prefill.py',s/'coalesced_body_block_shader.h',","s/'run_packed_body_prefill.py',s/'run_wide_body_decode.py',s/'coalesced_body_block_shader.h',")
exec(compile(source,__file__,'exec'))
