"""Current large-body decode versus added smaller matrices; fixed64 threads."""
from pathlib import Path

wrapper=Path(__file__).with_name('run_packed_body_prefill.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('from make_packed_body_prefill import context as generate_context,header as generate_header',
                      'from make_small_body_threshold import context as generate_context,header as generate_header,harness as generate_harness')
source=source.replace('build-interleaved-bf16-body','build-coalesced-bf16-body').replace('interleaved-body-native-checks-20261015','coalesced-body-native-checks-20261015')
source=source.replace("s/'make_packed_body_prefill.py'","s/'make_small_body_threshold.py',s/'audit_small_body_threshold.py',s/'audit_coalesced_body_block.py',s/'run_packed_body_prefill.py',s/'coalesced_body_block_gpu.mm',root/'source/src/backends/metal/coalesced_body_block_shader.h'")
source=source.replace("cpp.write_bytes((s/'packed_body_prefill_gpu.mm').read_bytes())", "cpp.write_bytes(generate_harness((s/'coalesced_body_block_gpu.mm').read_text()).encode())")
source=source.replace('packed_body_prefill_gpu','small_body_threshold_gpu').replace('metal_context_packed_prefill','metal_context_small_body_threshold')
source=source.replace("s/'small_body_threshold_gpu.mm'","s/'coalesced_body_block_gpu.mm'")
source=source.replace("rows=80*batch if model=='Qwen3.5-0.8B' else (24 if batch==1 else 44) if model=='Qwen3-0.6B' else 26*batch",'rows=batch')
source=source.replace('len(checks)==24','len(checks)==8').replace('for policy in (1,2,3):','for policy in (1,):')
needle='paths=[Path(__file__),';assert source.count(needle)==1
source=source.replace(needle,"run('cpu-admission',[sys.executable,str(s/'audit_small_body_threshold.py'),str(root),str(out/'admission')])\n"+needle)
needle='for p in paths:';at=source.index(needle)
source=source[:at]+"metadata.update(scope='Private current coalesced large-body decode versus added524288-value admission, fixed64 threads. All original projection workloads, bits and FP32 arithmetic unchanged.',limitations='Both policies retain current coalesced large matrices; only candidate enables smaller layouts. Extra buffers/packing exist outside timed windows for both. Independent projection sets, not full-model TPS/TTFT or memory evidence.')\n"+source[at:]
needle="assert '-DGARNET_METAL_INTERLEAVED_BF16_BODY=1' in defines";assert source.count(needle)==1
source=source.replace(needle,needle+"\nassert '-DGARNET_METAL_COALESCED_BF16_BODY=1' in defines")
exec(compile(source,__file__,'exec'))
