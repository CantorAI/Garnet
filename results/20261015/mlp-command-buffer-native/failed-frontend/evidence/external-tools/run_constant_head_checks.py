"""Validate optional batch-one constant-address heads against retained kernels."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_direct_state_zero_checks.py').read_text()
source=source.replace('build-direct-state-zero','build-constant-bf16-head').replace('METAL_DIRECT_STATE_ZERO','METAL_CONSTANT_BF16_HEAD')
destination=Path(sys.argv[2]).with_suffix('.wrapper.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
