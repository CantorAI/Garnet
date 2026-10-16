"""Original nineteen gates plus three chunk-copy admission regression stages."""
from pathlib import Path
import sys
folder=Path(__file__).parent;ns={'__file__':__file__}
wrapper=(folder/'run_guarded_global_half_alpha_checks.py').read_text();exec(compile(wrapper[:wrapper.index('\ndestination=')],__file__,'exec'),ns);source=ns['source']
source=source.replace('build-global-half-alpha','build-dirty-prefix-reset-repaired').replace("'METAL_GLOBAL_HALF_ALPHA'","'METAL_GLOBAL_HALF_ALPHA','METAL_DIRTY_PREFIX_RESET'")
needle="run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])"
assert source.count(needle)==1
source=source.replace(needle,needle+"\nfor model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):\n    run(model+'-prefix-copy',[py,str(s/'test_prefix_reset_prefill_reference.py'),runtime,str(build),str(r/'source'),str(models/model),str(output/(model+'-prefix-copy-fixtures'))])")
source=source.replace('assert len(statuses)==19','assert len(statuses)==22')
source=source.replace("'test2026/metal/run_joined_prefill_checks.py')}","'test2026/metal/run_joined_prefill_checks.py','test2026/metal/run_prefix_reset_prefill_reference.py','test2026/metal/test_prefix_reset_prefill_reference.py')}")
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode());exec(compile(source,str(destination),'exec'))
