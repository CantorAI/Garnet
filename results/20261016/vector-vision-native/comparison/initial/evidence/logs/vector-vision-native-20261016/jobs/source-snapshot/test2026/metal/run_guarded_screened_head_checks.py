"""Fresh screened-head build, its numerical/ownership CTest and all original gates."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_global_half_alpha_checks.py').read_text();needle="destination=Path(sys.argv[2]).with_suffix";assert source.count(needle)==1
addition='''source=source.replace('build-global-half-alpha','build-screened-head').replace("'METAL_GLOBAL_HALF_ALPHA'","'METAL_GLOBAL_HALF_ALPHA','METAL_SCREENED_HEAD'")
source=source.replace("run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])", "run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])\\nrun('screened-real-heads',[py,str(s/'test_screened_head_real.py'),str(r),str(output/'trained-heads'),str(build)])")
source=source.replace('assert len(statuses)==19','assert len(statuses)==20')
'''
source=source.replace(needle,addition+needle)
exec(compile(source,__file__,'exec'))
