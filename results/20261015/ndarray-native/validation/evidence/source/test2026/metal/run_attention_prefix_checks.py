"""Validate changed chunk attention: standard handoff, exact tails and 80-slot VL."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_joined_prefill_checks.py').read_text()
assert source.count("build=r/'build-joined-prefill'")==1
source=source.replace("build=r/'build-joined-prefill'","build=r/'build-attention-prefix'")
needle="'METAL_RESIDENT_KV_INPLACE','METAL_JOINED_PREFILL'"
assert source.count(needle)==1
source=source.replace(needle,"'METAL_RESIDENT_KV_INPLACE','METAL_ATTENTION_PREFIX'")
start=source.index("for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):")
end=source.index('tasks=',start)
source=source[:start]+source[end:]
needle="(output/'CTest-LastTest.log').write_bytes"
source=source.replace(needle,"run('qwen35-extended-prefill',[py,str(s/'test_qwen35_extended_prefill_reference.py'),runtime,str(build),str(r/'source'),str(models/'Qwen3.5-0.8B')])\n"+needle)
source=source.replace('assert len(statuses)==18','assert len(statuses)==16')
source=source.replace('All standard/native-pair stages, 48 exact-tail comparisons and three private KV suites passed','All attention-prefix handoff, exact-tail and extended-prefill stages passed')
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
