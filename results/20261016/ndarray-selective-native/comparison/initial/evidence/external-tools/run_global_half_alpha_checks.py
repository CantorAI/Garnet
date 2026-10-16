"""Validate lossless common-scale MPS alpha against retained native policy."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_direct_state_zero_checks.py').read_text()
source=source.replace('build-direct-state-zero','build-global-half-alpha').replace('METAL_DIRECT_STATE_ZERO','METAL_GLOBAL_HALF_ALPHA')
source=source.replace('All direct-state-zero checks','All global-half-alpha checks')
destination=Path(sys.argv[2]).with_suffix('.wrapper.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
