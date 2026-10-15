"""Actual learned-value component test; no full-model or comparator speed claim."""
from pathlib import Path
import subprocess,sys
folder=Path(__file__).parent
subprocess.run([sys.executable,str(folder/'make_coalesced_activation_decode.py'),str(folder/'coalesced_activation_decode_gpu.mm'),str(folder/'coalesced_activation_decode.metal')],check=True)
source=(folder/'run_joined_activation_decode_probe.py').read_text().replace('joined_activation_decode_gpu.mm','coalesced_activation_decode_gpu.mm').replace('joined_activation_decode.metal','coalesced_activation_decode.metal').replace('joined_activation_decode_gpu','coalesced_activation_decode_gpu')
source=source.replace('Joined lossless input components versus native BF16 decode; component only, no serving or vLLM claim','Coalesced mixed-input/FP32-accumulator lossless components versus native BF16; component only')
source=source.replace('MPS FP32 result','Explicit FP32 SIMD matrix accumulator')
source=source.replace('paths=[native,',"paths=[s/'make_coalesced_activation_decode.py',s/'coalesced_activation_product.metal',s/'joined_activation_decode_gpu.mm',s/'joined_activation_decode.metal',s/'run_joined_activation_decode_probe.py',native,")
exec(compile(source,__file__,'exec'))
