"""Verify extra packed prefill chunks, including the frozen text-B1 chunk 24."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_prefill_residual_norm_probe.py').read_text()
source=source.replace("str(shader),str(probe)])","str(shader),str(probe),'--extra-chunks'])")
destination=Path(sys.argv[2]).with_suffix('.wrapper.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
