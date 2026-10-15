"""Fresh global-alpha build with complete original cases and guarded retry groups."""
import ast,sys
from pathlib import Path
folder=Path(__file__).parent;ns={'__file__':__file__,'__doc__':__doc__}
wrapper=(folder/'run_global_half_alpha_checks.py').read_text();exec(wrapper[:wrapper.index('destination=')],ns)
wrapper=ns['source'];exec(wrapper[:wrapper.index('destination=')],ns);source=ns['source']
tree=ast.parse((folder/'resume_packed_prefill_norm_checks.py').read_text())
retry=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='retry' for t in n.targets))
assert source.count('def run(name,command):')==1
source=source.replace('def run(name,command):','def run_once(name,command):')
needle="assert not build.exists(),'Keep existing build artifacts; use a fresh candidate directory'";assert source.count(needle)==1
source=source.replace(needle,retry+'\n'+needle)
source=source.replace('build-global-half-alpha','build-shape-selected-two-body').replace("'METAL_GLOBAL_HALF_ALPHA'","'METAL_GLOBAL_HALF_ALPHA','METAL_SCREENED_HEAD','METAL_PREFILL_STREAM','METAL_DIRTY_PREFIX_RESET','METAL_SERIAL_VISION_PREFETCH','METAL_SMALL_SCREEN_SEED','METAL_INTERLEAVED_SCREEN_INDEX','METAL_QWEN35_WIDE_BATCH_PREFILL','METAL_INTERLEAVED_BF16_BODY','METAL_COALESCED_BF16_BODY'")
source=source.replace("run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])", "run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])\nrun('vision-full-grid',[py,str(s/'test_serial_vision_grid.py'),str(r),str(output/'full-grid'),str(build)])\nrun('screened-real-heads',[py,str(s/'test_screened_head_real.py'),str(r),str(output/'trained-heads'),str(build)])")
source=source.replace('assert len(statuses)==19','assert len(statuses)==21')
source += """
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for label,helper in [('prefix','test_prefix_reset_session_reference.py'),('screened','test_interleaved_head_session_reference.py')]:
        run(model+'-'+label+'-recovery',[sys.executable,str(s/helper),runtime,str(build),str(r/'source'),str(r/'models'/model),str(output/(label+'-recovery-'+model))])
assert len(statuses)==27 and all(v['exit_code']==0 for v in statuses)
print('Original19, exact trained vision, trained heads and both recovery suites PASS',flush=True)
"""
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))

run('wide-batch-trained-reference',[sys.executable,str(s/'test_wide_batch_prefill_reference.py'),runtime,str(build),str(r/'source'),str(r/'models/Qwen3.5-0.8B'),str(output/'wide-reference')])
assert len(statuses)==28
