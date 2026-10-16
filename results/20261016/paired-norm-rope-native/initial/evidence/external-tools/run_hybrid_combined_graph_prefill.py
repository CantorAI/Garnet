"""One native special-key submission plus combined graphs, all original matrices."""
from pathlib import Path

wrapper=Path(__file__).with_name('run_combined_graph_prefill.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
needle='paths=[Path(__file__),';assert source.count(needle)==1
source=source.replace(needle,"probe=out/'hybrid_combined_graph_prefill_gpu.mm'\nrun('harness',[sys.executable,str(s/'make_hybrid_combined_graph_prefill.py'),str(probe)])\n"+needle+"s/'make_hybrid_combined_graph_prefill.py',s/'run_combined_graph_prefill.py',")
source=source.replace("str(s/'combined_graph_prefill_gpu.mm')","str(probe)")
source=source.replace("binary=out/'combined_graph_prefill_gpu'","binary=out/'hybrid_combined_graph_prefill_gpu'")
needle="(out/'metadata.json').write_bytes";at=source.index(needle)
source=source[:at]+"metadata.update(scope='Complete independent first-layer projection sets with native original special-key fallback plus FP32 placeholder/constant graphs for all remaining matrices. Original workloads and bounds unchanged.',limitations='Candidate wall clocks include the native key submission and synchronous graph call. Initial allocations/uploads/compile excluded from warm windows. First candidate call includes native key work and graph setup. No complete-layer dependency graph, full-model TTFT or vLLM ratio.',generated_cpp_sha256=digest(probe))\n"+source[at:]
exec(compile(source,__file__,'exec'))
