"""All six native matched prefill profiles for lossless global MPS alpha."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_packed_prefill_norm_profiles.py').read_text().replace('compare_packed_prefill_norm.py','compare_global_half_alpha.py')
destination=Path(sys.argv[2]).with_suffix('.outer.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
