"""Audit complete hybrid projection workloads, original bounds and timing scope."""
from pathlib import Path

source=Path(__file__).with_name('verify_combined_graph_prefill.py').read_text()
source=source.replace("assert len(stages)==4 and {s['name'] for s in stages}=={'compile'}", "assert len(stages)==5 and {s['name'] for s in stages}=={'harness','compile'}")
needle="    assert meta['reference_static_library_sha256']==";at=source.index(needle)
source=source[:at]+"    assert meta['generated_cpp_sha256']==digest(out/'hybrid_combined_graph_prefill_gpu.mm')\n"+source[at:]
needle="                    assert check['fp32_activation_weight_product_output']";assert source.count(needle)==1
source=source.replace(needle,needle+'''
                    expected_keys=sum(c['tensor'].startswith('model.layers.') and c['tensor'].endswith('.self_attn.k_proj.weight') for c in assets['cases'])
                    assert expected_keys==(0 if model=='Qwen3.5-0.8B' else 1)
                    assert check['native_key_fallbacks']==expected_keys and check['graph_projections']==len(assets['cases'])-expected_keys
                    assert check['native_key_bit_exact']
''')
source=source.replace('All independent first-layer projections in one native submission or one graph call.', 'All original first-layer projections: candidate includes native exact-key work plus graph call; no omitted matrices.')
exec(compile(source,__file__,'exec'))
