"""Current scalar prefetch8 versus float2/float4 dimensions plus same-pipeline null."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_serial_vision_context.py').read_text()
source=source.replace('from make_serial_vision_context import generate','from make_vector_vision_context import generate')
source=source.replace("s/'make_serial_vision_context.py',","s/'make_vector_vision_context.py',s/'make_serial_vision_context.py',s/'run_serial_vision_context.py',root/'source/src/backends/metal/serial_vision_prefetch_shader.h',")
old="generate(native.read_text(),(s/'vision_fused_context_gpu.mm').read_text())";assert source.count(old)==1
source=source.replace(old,"generate(native.read_text(),(s/'vision_fused_context_gpu.mm').read_text(),(root/'source/src/backends/metal/serial_vision_prefetch_shader.h').read_text())")
source=source.replace("variants=['original-tg64','original-tg128','original-tg256','ordered-prefetch4-tg128','ordered-prefetch8-tg128']","variants=['same-scalar-prefetch8-null','ordered-vector2-prefetch8','ordered-vector4-prefetch8']")
source=source.replace('serial-vision.metal','vector-vision.metal').replace('serial_vision_context_gpu','vector_vision_context_gpu')
source=repr(__doc__)+source[source.index('\n'):]
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,__file__,'exec'))
