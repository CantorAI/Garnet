"""Private compact eight-output decode versus current compact four-output kernel."""
from pathlib import Path

p=Path(__file__).with_name('run_packed_body_prefill.py');wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns);source=ns['source']
source=source.replace('from make_packed_body_prefill import context as generate_context,header as generate_header',
                      'from make_coalesced_eight_body import context as generate_context,header as generate_header,harness as generate_harness')
source=source.replace('build-interleaved-bf16-body','build-coalesced-bf16-body').replace('interleaved-body-native-checks-20261015','coalesced-body-native-checks-20261015')
source=source.replace("s/'make_packed_body_prefill.py'","s/'make_coalesced_eight_body.py',s/'coalesced_eight_body_shader.h',s/'make_unrolled_coalesced_eight_shader.py',s/'run_packed_body_prefill.py',s/'coalesced_body_block_gpu.mm',root/'source/src/backends/metal/coalesced_body_block_shader.h',root/'source/src/backends/metal/mlp_prefill_graph_impl.h'")
source=source.replace("cpp.write_bytes((s/'packed_body_prefill_gpu.mm').read_bytes())", "cpp.write_bytes(generate_harness((s/'coalesced_body_block_gpu.mm').read_text()).encode())\n(out/'coalesced_eight_body_shader.h').write_bytes((s/'coalesced_eight_body_shader.h').read_bytes())")
source=source.replace('packed_body_prefill_gpu','coalesced_eight_body_gpu').replace('metal_context_packed_prefill','metal_context_coalesced_eight')
source=source.replace("s/'coalesced_eight_body_gpu.mm'","s/'coalesced_body_block_gpu.mm'")
source=source.replace("rows=80*batch if model=='Qwen3.5-0.8B' else (24 if batch==1 else 44) if model=='Qwen3-0.6B' else 26*batch",'rows=batch')
needle="assert '-DGARNET_METAL_INTERLEAVED_BF16_BODY=1' in defines";assert source.count(needle)==1
source=source.replace(needle,needle+"\nassert '-DGARNET_METAL_COALESCED_BF16_BODY=1' in defines and '-DGARNET_METAL_MLP_PREFILL_GRAPH=1' not in defines")
needle='for p in paths:';at=source.index(needle)
source=source[:at]+"metadata.update(scope='Private current compact four-output decode versus eight-output reuse, unchanged exact BF16 payload and FP32 accumulation order. Candidate64/128/256 threads; control always64. No full-model TPS/TTFT or vLLM result.',limitations='Independent trained projection sets. Common setup/packing outside timing; no extra weight layout and no memory saving claim. Original numerical/bitwise/safety gates unchanged.')\n"+source[at:]
source=repr(__doc__)+source[source.index('\n'):]
exec(compile(source,__file__,'exec'))
