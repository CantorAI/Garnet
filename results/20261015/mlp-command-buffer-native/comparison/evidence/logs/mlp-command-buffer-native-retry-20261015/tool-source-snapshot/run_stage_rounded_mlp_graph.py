"""Dependent MLP with FP64 dots rounded at existing FP32 reference boundaries."""
from pathlib import Path

wrapper=Path(__file__).with_name('run_dependent_mlp_graph.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
needle='paths=[Path(__file__),';assert source.count(needle)==1
source=source.replace(needle,"probe=out/'stage_rounded_mlp_graph_gpu.mm'\nrun('harness',[sys.executable,str(s/'make_stage_rounded_mlp_graph.py'),str(probe)])\n"+needle+"s/'make_stage_rounded_mlp_graph.py',s/'run_dependent_mlp_graph.py',s/'reference_dense.py',")
source=source.replace("str(s/'dependent_mlp_graph_gpu.mm')","str(probe)").replace("binary=out/'dependent_mlp_graph_gpu'","binary=out/'stage_rounded_mlp_graph_gpu'")
needle="(out/'metadata.json').write_bytes";at=source.index(needle)
source=source[:at]+"metadata.update(scope='Dependent MLP, existing FP32-valued reference semantics: FP64 dots rounded toFP32 at each projection, FP32 activation, finalFP32 projection. Graph/native operations unchanged from prior stress test.',limitations='One native submission or synchronous graph call perMLP; VL blocks remain separate and sequential. Existing FP32 stage reference uses wider CPU dot accumulation. First/setup costs separate; no full-model or vLLM result.',numerical_bounds='Unchanged5e-6+5e-5*abs(stage-rounded reference), and unchanged5e-6+5e-5*abs(native). Prior whole-chain FP64 stress results remain separately archived.',generated_cpp_sha256=digest(probe))\n"+source[at:]
exec(compile(source,__file__,'exec'))
