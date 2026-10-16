"""All six native residual-norm profiles with frozen prompts and complete outputs."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_constant_head_profiles.py').read_text()
source=source.replace('compare_constant_head.py','compare_residual_norm.py').replace('for batch in (1,):','for batch in (1,2):').replace('assert len(profiles)==3','assert len(profiles)==6')
assert '--confirm' not in sys.argv[3:]
destination=Path(sys.argv[2]).with_suffix('.wrapper.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
