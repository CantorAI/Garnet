"""Validate optional three-output packed prefill fusion against retained kernels."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_direct_state_zero_checks.py').read_text()
source=source.replace('build-direct-state-zero','build-packed-prefill-norm').replace('METAL_DIRECT_STATE_ZERO','METAL_PACKED_PREFILL_NORM')
source=source.replace('All direct-state-zero checks','All packed-prefill-norm checks')
destination=Path(sys.argv[2]).with_suffix('.wrapper.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
