"""Build and validate direct state reset with the retained attention policy."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_joined_prefill_checks.py').read_text()
assert source.count("build=r/'build-joined-prefill'")==1
source=source.replace("build=r/'build-joined-prefill'","build=r/'build-decode-subsimd'")
needle="'METAL_RESIDENT_KV_INPLACE','METAL_JOINED_PREFILL'"
assert source.count(needle)==1
source=source.replace(needle,"'METAL_RESIDENT_KV_INPLACE','METAL_ATTENTION_PREFIX','METAL_SUBSIMD_DECODE'")
needle="(output/'CTest-LastTest.log').write_bytes"
source=source.replace(needle,"run('qwen35-extended-prefill',[py,str(s/'test_qwen35_extended_prefill_reference.py'),runtime,str(build),str(r/'source'),str(models/'Qwen3.5-0.8B')])\n"+needle)
source=source.replace('assert len(statuses)==18','assert len(statuses)==19')
source=source.replace('All standard/native-pair stages, 48 exact-tail comparisons and three private KV suites passed','All direct-state-zero checks, exact tails and private KV suites passed')
source=source.replace('def guard(label):\n    strict_guard()', '''def guard(label):
    import shutil
    if shutil.disk_usage(r).free<2*2**30:raise OSError('Disk free space below 2 GiB')
    strict_guard()''')
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
