"""Audit existing reference boundaries, original tolerances and dependent dataflow."""
from pathlib import Path

wrapper=Path(__file__).with_name('verify_dependent_mlp_graph.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("assert len(stages)==4 and {s['name'] for s in stages}=={'compile'}", "assert len(stages)==5 and {s['name'] for s in stages}=={'harness','compile'}")
needle="    assert meta['reference_static_library_sha256']==";at=source.index(needle)
source=source[:at]+'''    assert meta['generated_cpp_sha256']==digest(out/'stage_rounded_mlp_graph_gpu.mm')
    original=(snap/'test2026/metal/dependent_mlp_graph_gpu.mm').read_text()
    expected=original.replace('std::vector<double> activated(p.intermediate);','std::vector<float> activated(p.intermediate);').replace('activated[j]=(g/(1.+std::exp(-g)))*u;','float gate=static_cast<float>(g),up=static_cast<float>(u);activated[j]=(gate/(1.f+std::exp(-gate)))*up;').replace('p.reference[r*p.width+c]=sum;p.bounds[r*p.width+c]=5e-6+5e-5*std::abs(sum);','float rounded=static_cast<float>(sum);p.reference[r*p.width+c]=rounded;p.bounds[r*p.width+c]=5e-6+5e-5*std::abs(double(rounded));')
    assert (out/'stage_rounded_mlp_graph_gpu.mm').read_bytes()==expected.encode()
    reference=(snap/'test2026/metal/reference_dense.py').read_text()
    assert 'def project(x,w):return (x.astype(np.float64)@w.astype(np.float64).T).astype(np.float32)' in reference
    assert 'activated=gate/(1+np.exp(-gate))*up' in reference
'''+source[at:]
source=source.replace('Full CPU FP64 trained-layer envelope', 'Existing FP32 stage boundaries with wider CPU dot accumulation and unchanged trained-layer envelope')
exec(compile(source,__file__,'exec'))
