"""Exclusive packed prefill Add/Layout/RMS probe; preserve all three outputs."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_residual_norm_probe.py').read_text()
source=source.replace("paths=[native,s/'residual_norm_gpu.mm',Path(__file__)","paths=[native,s/'residual_norm_gpu.mm',s/'make_prefill_residual_norm_probe.py',s/'run_residual_norm_probe.py',Path(__file__)")
# Replace only shader generation. The canonical trained-weight extraction,
# checkpoint hashes and exclusive process guard remain unchanged.
start=source.index('text=native.read_text()');end=source.index('cases=[]',start)
source=source[:start]+"shader=out/'prefill-residual-norm.metal';probe=out/'prefill_residual_norm_gpu.mm'\nrun('generate',[sys.executable,str(s/'make_prefill_residual_norm_probe.py'),str(native),str(shader),str(probe)])\nmetadata['shader_sha256']=digest(shader)\n"+source[end:]
source=source.replace("binary=out/'residual_norm_gpu'","binary=out/'prefill_residual_norm_gpu'").replace("str(s/'residual_norm_gpu.mm')","str(probe)")
source=source.replace("print('Residual add and RMS fused component passed exact fresh-input, both-output, padding and FP64 gates'","print('Packed residual, layout and RMS component passed exact fresh-input, three-output, padding and FP64 gates'")
source=source.replace('"""Exclusive residual-add/RMS fusion, original float math and both outputs."""',repr(__doc__))
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
