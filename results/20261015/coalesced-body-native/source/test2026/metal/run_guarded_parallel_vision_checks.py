"""Original nineteen gates plus actual trained full-grid embeddings and dispatch checks."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_global_half_alpha_checks.py').read_text();needle="destination=Path(sys.argv[2]).with_suffix";assert source.count(needle)==1
addition='''source=source.replace('build-global-half-alpha','build-parallel-vision-context').replace("'METAL_GLOBAL_HALF_ALPHA'","'METAL_GLOBAL_HALF_ALPHA','METAL_FUSED_VISION_CONTEXT','METAL_PARALLEL_VISION_CONTEXT'")
source=source.replace("(output/'CTest-LastTest.log').write_bytes", "run('vision-full-grid',[py,str(s/'test_parallel_vision_grid.py'),str(r),str(output/'full-grid'),str(build)])\\n"+"(output/'CTest-LastTest.log').write_bytes")
source=source.replace('assert len(statuses)==19','assert len(statuses)==20')
'''
source=source.replace(needle,addition+needle)
exec(compile(source,__file__,'exec'))
