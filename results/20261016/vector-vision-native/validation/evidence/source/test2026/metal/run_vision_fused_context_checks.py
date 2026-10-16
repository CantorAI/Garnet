"""Reuse the standard eighteen sequential handoff stages for the fused candidate."""
from pathlib import Path
import sys
script=Path(__file__).with_name('run_joined_prefill_checks.py')
source=script.read_text()
assert source.count("build=r/'build-joined-prefill'")==1
source=source.replace("build=r/'build-joined-prefill'","build=r/'build-vision-fused-context'")
source=source.replace("'METAL_RESIDENT_KV_INPLACE','METAL_JOINED_PREFILL'","'METAL_RESIDENT_KV_INPLACE','METAL_FUSED_VISION_CONTEXT'")
source=source.replace('All standard/native-pair stages, 48 exact-tail comparisons and three private KV suites passed','All fused-vision handoff stages, 48 exact-tail comparisons and three private KV suites passed')
destination=Path(sys.argv[2]).with_suffix('.executed.py')
destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
