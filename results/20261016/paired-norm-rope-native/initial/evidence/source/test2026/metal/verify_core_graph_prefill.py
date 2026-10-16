"""Audit explicit core compilation settings and all original numerical profiles."""
from pathlib import Path

source=Path(__file__).with_name('verify_combined_graph_prefill.py').read_text()
source=source.replace("assert len(stages)==4 and {s['name'] for s in stages}=={'compile'}", "assert len(stages)==5 and {s['name'] for s in stages}=={'harness','compile'}")
needle="    assert meta['reference_static_library_sha256']==";at=source.index(needle)
source=source[:at]+"    assert meta['generated_cpp_sha256']==digest(out/'core_graph_prefill_gpu.mm')\n"+source[at:]
needle="                    assert check['fp32_activation_weight_product_output']";assert source.count(needle)==1
source=source.replace(needle,needle+"\n                    assert check['graph_optimization_level']==0 and check['graph_wait_until_completed']")
source=source.replace('Original FP32 tensor values; placeholder/constant weight policies.', 'Original FP32 tensor values; placeholder/constant policies, explicit core compilation and synchronous execution descriptor.')
exec(compile(source,__file__,'exec'))
