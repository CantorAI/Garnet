"""Core-only graph compilation, including original special keys, unchanged bounds."""
from pathlib import Path

wrapper=Path(__file__).with_name('run_combined_graph_prefill.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
needle='paths=[Path(__file__),';assert source.count(needle)==1
source=source.replace(needle,"probe=out/'core_graph_prefill_gpu.mm'\nrun('harness',[sys.executable,str(s/'make_core_graph_prefill.py'),str(probe)])\n"+needle+"s/'make_core_graph_prefill.py',s/'run_combined_graph_prefill.py',")
source=source.replace("str(s/'combined_graph_prefill_gpu.mm')","str(probe)").replace("binary=out/'combined_graph_prefill_gpu'","binary=out/'core_graph_prefill_gpu'")
needle="(out/'metadata.json').write_bytes";at=source.index(needle)
source=source[:at]+"metadata.update(scope='Complete original projection sets, including special keys, with explicit MPSGraphOptimizationLevel0 and synchronous execution descriptor. Original FP32 values and bounds unchanged.',limitations='Only core compilation passes requested; explicit Metal command queue. Candidate API is runAsync with waitUntilCompleted=YES. No native key fallback, matrix omission or tolerance relaxation. Original first graph/setup costs separate; component wall clocks only.',generated_cpp_sha256=digest(probe))\n"+source[at:]
exec(compile(source,__file__,'exec'))
