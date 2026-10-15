"""Audit exact compact-eight math, route assertions, safety and admitted timing."""
from pathlib import Path
p=Path(__file__).with_name('verify_coalesced_body_block.py');wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns);source=ns['source']
for old,new in [('make_coalesced_body_block','make_coalesced_eight_body'),('metal_context_coalesced_block','metal_context_coalesced_eight'),('coalesced_body_block_gpu','coalesced_eight_body_gpu')]:source=source.replace(old,new)
needle="assert harness.read_bytes()==(snap/'test2026/metal/coalesced_eight_body_gpu.mm').read_bytes()";assert source.count(needle)==1
source=source.replace(needle,"assert harness.read_bytes()==factory['harness']((snap/'test2026/metal/coalesced_body_block_gpu.mm').read_text()).encode()\n    assert (out/'coalesced_eight_body_shader.h').read_bytes()==(snap/'test2026/metal/coalesced_eight_body_shader.h').read_bytes()\n    assert '-DGARNET_METAL_COALESCED_BF16_BODY=1' in meta['context_macro_defines'] and '-DGARNET_METAL_MLP_PREFILL_GRAPH=1' not in meta['context_macro_defines']")
source=source.replace('Private coalesced exact12-bit BF16 blocks with uniform native fallback. Independent CPU block coverage and runtime counts, exact native output bits and original FP64 envelope; tail, ownership and recovery controls. Component only; no model TPS/TTFT or vLLM claim.',
                      'Private eight-output reuse of unchanged compact four-output blocks. Control always current compact64-thread kernel; candidate64/128/256 threads. Exact native output bits, original FP64 envelope and ownership/recovery tests; actual eight-route counters asserted by frozen harness. Component only; no complete-model or vLLM claim.')
exec(compile(source,__file__,'exec'))
