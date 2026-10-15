"""Exclusive learned first-layer prefill component study; no inference speed claim."""
from pathlib import Path
import subprocess,sys
folder=Path(__file__).parent
subprocess.run([sys.executable,str(folder/'make_coalesced_activation_prefill.py'),str(folder/'coalesced_activation_prefill_gpu.mm'),str(folder/'coalesced_activation_prefill.metal')],check=True)
source=(folder/'run_joined_activation_decode_probe.py').read_text().replace('joined_activation_decode_gpu.mm','coalesced_activation_prefill_gpu.mm').replace('joined_activation_decode.metal','coalesced_activation_prefill.metal').replace('joined_activation_decode_gpu','coalesced_activation_prefill_gpu')
source=source.replace('Joined lossless input components versus native BF16 decode; component only, no serving or vLLM claim','Lossless coalesced prefill FP32 MMA versus selected global-alpha MPS; component only, synthetic FP32 inputs and actual first-layer weights')
source=source.replace('Original native BF16 shaders with default Metal options','Selected global-alpha MPS: FP32 input, lossless half weight storage and FP32 output').replace('MPS FP32 result','Explicit FP32 SIMD matrix accumulator')
source=source.replace('paths=[native,',"paths=[s/'make_coalesced_activation_prefill.py',s/'make_coalesced_activation_decode.py',s/'coalesced_activation_product.metal',s/'joined_activation_decode_gpu.mm',s/'joined_activation_decode.metal',s/'run_joined_activation_decode_probe.py',s/'prepare_prefill_tile_probe.py',native,")
needle=" assets=root/'logs/mps-gemv-probe-20261013/jobs'/(model+'-assets');info="
assert source.count(needle)==1
source=source.replace(needle," assets=out/(model+'-assets');run(model+'-prepare',[sys.executable,str(s/'prepare_prefill_tile_probe.py'),str(root/'models'/model/'model.safetensors'),str(assets)]);info=")
source=source.replace("assert len(checks)==2*(len(info['cases'])-len(excluded))","assert len(checks)==4*(len(info['cases'])-len(excluded))")
exec(compile(source,__file__,'exec'))
