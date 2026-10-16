"""Full original nineteen gates plus both optional CTests and three recovery suites."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_global_half_alpha_checks.py').read_text()
needle="destination=Path(sys.argv[2]).with_suffix"
assert source.count(needle)==1
extra="""source=source.replace('build-global-half-alpha','build-combined-prefill-reset').replace("'METAL_GLOBAL_HALF_ALPHA'","'METAL_GLOBAL_HALF_ALPHA','METAL_PREFILL_STREAM','METAL_DIRTY_PREFIX_RESET'")
source += '''
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    run(model+'-session-recovery',[sys.executable,str(s/'test_prefix_reset_session_reference.py'),runtime,str(build),str(r/'source'),str(r/'models'/model),str(output/('recovery-'+model))])
assert len(statuses)==22 and all(v['exit_code']==0 for v in statuses)
print('Combined prefill/reset original19 and recovery3 PASS',flush=True)
'''
"""
source=source.replace(needle,extra+needle)
exec(compile(source,__file__,'exec'))
