"""Sequential real-weight FP32 GEMV component with frozen numerical bounds."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_decode_subsimd_probe.py').read_text()
source=source.replace("s/'make_decode_subsimd_probe.py'", "s/'make_mps_gemv_probe.py'")
source=source.replace('decode_subsimd_gpu','mps_gemv_gpu').replace('len(info)*4','len(info)*2')
source=source.replace("'-framework','Metal',str(probe)","'-framework','Metal','-framework','MetalPerformanceShaders',str(probe)")
source=source.replace("scope='Real BF16 learned values and FP32 math; changed reduction order; component only'", "scope='MPS GEMV losslessly expanded FP32 weights versus compact BF16 control; FP32 input/output; component only'")
needle="paths=[native,s/'decode_wide_outputs_gpu.mm'"
source=source.replace(needle,"paths=[native,s/'make_decode_subsimd_probe.py',s/'run_decode_subsimd_probe.py',s/'decode_wide_outputs_gpu.mm'")
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
