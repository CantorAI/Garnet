"""Retain original MLP numerical/timing gates and check every GPU-cast weight bit."""
from pathlib import Path
p=Path(__file__).with_name('verify_stage_rounded_mlp_graph.py');wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns);source=ns['source']
source=source.replace('stage_rounded_mlp_graph_gpu','bf16_feed_mlp_graph_gpu')
start=source.index("    original=(snap/'test2026/metal/dependent_mlp_graph_gpu.mm').read_text()")
end=source.index("    reference=(snap/'test2026/metal/reference_dense.py').read_text()",start)
source=source[:start]+'''    import runpy
    factory=runpy.run_path(str(snap/'test2026/metal/make_bf16_feed_mlp_component.py'))
    expected=factory['generate']((snap/'test2026/metal/dependent_mlp_graph_gpu.mm').read_text())
    assert (out/'bf16_feed_mlp_graph_gpu.mm').read_bytes()==expected.encode()
    assert 'GPU BF16 cast changed original learned bits' in expected
    assert 'need(weights[i].dataType==MPSDataTypeFloat32' in expected
'''+source[end:]
old="assert all(v['kind'] in ('check','first_graph_run','samples') for v in values)";assert source.count(old)==1
source=source.replace(old,"""assert all(v['kind'] in ('check','first_graph_run','samples','weight_cast') for v in values)
        casts=[v for v in values if v['kind']=='weight_cast']
        learned={c['tensor']:c['rows']*c['inner'] for c in assets['cases'] if '.mlp.' in c['tensor']}
        assert len(casts)==len(learned) and {v['tensor'] for v in casts}==set(learned)
        assert all(v['bitwise_equal'] and v['values']==learned[v['tensor']] for v in casts)""")
source=source.replace('placeholder/constant weight policies.', 'Original BF16 placeholder/constant storage, mandatory GPU learned-bit roundtrips and exact FP32 casts before FP32 arithmetic.')
exec(compile(source,__file__,'exec'))
