"""Native ABBA 16-lane candidate versus otherwise identical attention control."""
from pathlib import Path
import sys
source=Path(__file__).with_name('compare_direct_state_zero.py').read_text()
source=source.replace('build-direct-state-zero','build-decode-subsimd').replace('logs/direct-state-zero-checks-20261013','logs/decode-subsimd-checks-20261013')
source=source.replace("value['direct_state_zero_enabled']", "value['subsimd_decode_enabled']")
source=source.replace("Path(__file__).with_name('compare_joined_prefill.py')", "Path("+repr(str(Path(__file__).with_name('compare_joined_prefill.py')))+")")
destination=Path(sys.argv[2]).with_suffix('.wrapper.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
