"""Isolated native BF16 output-column layout, original B1/B2 and strict arithmetic."""
from pathlib import Path
wrapper=Path(__file__).with_name('run_streamed_direct_mps_body.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('from make_direct_mps_body import context as generate_context,harness as generate_harness','from make_interleaved_bf16_body import context as generate_context,harness as generate_harness,header as generate_header')
source=source.replace("paths=[Path(__file__),","paths=[Path(__file__),s/'make_interleaved_bf16_body.py',s/'run_streamed_direct_mps_body.py',")
needle="cpp.write_bytes((s/'streamed_direct_mps_body_gpu.mm').read_bytes())";assert source.count(needle)==1
source=source.replace(needle,"cpp.write_bytes(generate_harness((s/'streamed_direct_mps_body_gpu.mm').read_text()).encode())\nheader_cpp=out/'metal_context.h';header_cpp.write_bytes(generate_header((root/'source/src/backends/metal/metal_context.h').read_text()).encode())\nmetadata['private_header_sha256']=digest(header_cpp)")
source=source.replace("out/'streamed_direct_mps_body_gpu","out/'interleaved_bf16_body_gpu").replace("out/'metal_context_direct.mm'","out/'metal_context_interleaved_body.mm'")
source=source.replace('original-sized inputs; raw BF16 control and forced mixed-storage MPS GEMM in the same isolated context copy','original-sized inputs; original BF16 bytes and FP32 arithmetic, four-output interleaved raw-BF16 storage in the same isolated context copy; original master retained; no MPS decode substitution')
source=source.replace('assert len(checks)==6','assert len(checks)==24')
source=source.replace("c=next(v for v in checks if (v['batch'],v['policy'])==(batch,policy));", "relevant=[v for v in checks if (v['batch'],v['policy'])==(batch,policy)];assert len(relevant)==4 and {v['pattern'] for v in relevant}==set(range(4)) and all(v['passed'] and v['bitwise_equal'] for v in relevant)\n            c=next(v for v in relevant if v['pattern']==0);")
source=source.replace('Completed streamed native MPS component','Completed isolated interleaved BF16 body component')
exec(compile(source,__file__,'exec'))
