"""All original 19 gates plus deferred-clear ownership/recovery CTest."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_global_half_alpha_checks.py').read_text();needle="destination=Path(sys.argv[2]).with_suffix";assert source.count(needle)==1
extra="""source=source.replace('build-global-half-alpha','build-deferred-state-zero').replace("'METAL_GLOBAL_HALF_ALPHA'","'METAL_GLOBAL_HALF_ALPHA','METAL_DEFERRED_STATE_ZERO'")
source=source.replace("'src/backends/metal/metal_context.mm','cmake/GarnetMetal.cmake'", "'src/backends/metal/metal_context.mm','test2026/metal/deferred_state_zero_test.cpp','cmake/GarnetMetal.cmake'")
"""
source=source.replace(needle,extra+needle);exec(compile(source,__file__,'exec'))
