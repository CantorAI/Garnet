"""Fail fast on trained vision bits; original 19, trained head and recovery gates."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_global_half_alpha_checks.py').read_text();needle="destination=Path(sys.argv[2]).with_suffix";assert source.count(needle)==1
addition='''source=source.replace('build-global-half-alpha','build-serial-vision-prefetch').replace("'METAL_GLOBAL_HALF_ALPHA'","'METAL_GLOBAL_HALF_ALPHA','METAL_SCREENED_HEAD','METAL_PREFILL_STREAM','METAL_DIRTY_PREFIX_RESET','METAL_SERIAL_VISION_PREFETCH'")
source=source.replace("run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])", "run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])\\nrun('vision-full-grid',[py,str(s/'test_serial_vision_grid.py'),str(r),str(output/'full-grid'),str(build)])\\nrun('screened-real-heads',[py,str(s/'test_screened_head_real.py'),str(r),str(output/'trained-heads'),str(build)])")
source=source.replace('assert len(statuses)==19','assert len(statuses)==21')
source += """
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for label,helper in [('prefix','test_prefix_reset_session_reference.py'),('screened','test_screened_head_session_reference.py')]:
        run(model+'-'+label+'-recovery',[sys.executable,str(s/helper),runtime,str(build),str(r/'source'),str(r/'models'/model),str(output/(label+'-recovery-'+model))])
assert len(statuses)==27 and all(v['exit_code']==0 for v in statuses)
print('Original19, exact trained vision, trained heads and both recovery suites PASS',flush=True)
"""
'''
source=source.replace(needle,addition+needle)
exec(compile(source,__file__,'exec'))
