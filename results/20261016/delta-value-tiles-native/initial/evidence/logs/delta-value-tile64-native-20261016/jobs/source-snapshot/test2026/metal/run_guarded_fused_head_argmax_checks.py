"""All original 19 model gates and the fused head numeric/lifecycle CTest."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_global_half_alpha_checks.py').read_text()
needle="destination=Path(sys.argv[2]).with_suffix"
assert source.count(needle)==1
extra="""source=source.replace('build-global-half-alpha','build-fused-head-argmax').replace("'METAL_GLOBAL_HALF_ALPHA'","'METAL_GLOBAL_HALF_ALPHA','METAL_FUSED_HEAD_ARGMAX'")
source=source.replace("'src/backends/metal/metal_context.mm','cmake/GarnetMetal.cmake'", "'src/backends/metal/metal_context.mm','src/backends/metal/fused_head_argmax_shader.h','src/backends/metal/compiled_graph.h','test2026/metal/fused_head_argmax_test.cpp','cmake/GarnetMetal.cmake'")
"""
source=source.replace(needle,extra+needle)
exec(compile(source,__file__,'exec'))
