"""Complete gate/up-SiLU-down dependencies against current native MLP execution."""
from pathlib import Path

wrapper=Path(__file__).with_name('run_combined_graph_prefill.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('combined_graph_prefill_gpu.mm','dependent_mlp_graph_gpu.mm').replace("binary=out/'combined_graph_prefill_gpu'","binary=out/'dependent_mlp_graph_gpu'")
source=source.replace('paths=[Path(__file__),',"paths=[Path(__file__),s/'run_combined_graph_prefill.py',")
source=source.replace("v['projections']==len(info['cases'])","v['projections']==sum('.mlp.' in c['tensor'] for c in info['cases'])")
needle="(out/'metadata.json').write_bytes";at=source.index(needle)
source=source[:at]+"metadata.update(scope='Complete dependent MLP fixtures: native gate/up, SiLU multiplication, down projection versus corresponding FP32 graph. Text/coder layer0 and VL layers0/3; original checkpoint values.',limitations='One native submission or one synchronous graph call PER MLP. VL layers0/3 run sequentially through separate graphs, never fused across layers. True dependencies retained within each MLP. Placeholder/constant policies, first setup separate. Independent CPU FP64 complete MLP reference; no full-model TTFT or vLLM claim.',numerical_bounds='Original trained-layer envelope5e-6+5e-5*abs(full FP64 MLP reference), and unchanged5e-6+5e-5*abs(native). Failed native or candidate profiles have no timed windows.')\n"+source[at:]
exec(compile(source,__file__,'exec'))
