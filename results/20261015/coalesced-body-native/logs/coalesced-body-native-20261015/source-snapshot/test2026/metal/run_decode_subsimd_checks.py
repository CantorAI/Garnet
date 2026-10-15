"""Build and validate sub-SIMD decode; original numerical tolerances retained."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_direct_state_zero_checks.py').read_text()
source=source.replace('build-direct-state-zero','build-decode-subsimd').replace('METAL_DIRECT_STATE_ZERO','METAL_SUBSIMD_DECODE')
# Expand the existing wrapper, then resolve its final source with the new policy.
temporary=Path(sys.argv[2]).with_suffix('.wrapper.py');temporary.write_bytes(source.encode())
source=source.replace("Path(__file__).with_name('run_joined_prefill_checks.py')", "Path("+repr(str(Path(__file__).with_name('run_joined_prefill_checks.py')))+")")
exec(compile(source,str(temporary),'exec'))
