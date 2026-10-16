"""Freeze unchanged vision numerical/timing gates with exact selected-control regeneration."""
from pathlib import Path
s=Path(__file__).resolve().parent;source=(s/'verify_serial_vision_context.py').read_text()
source=source.replace('make_serial_vision_context.py','make_vector_vision_context.py').replace('serial-vision.metal','vector-vision.metal').replace('serial_vision_context_gpu','vector_vision_context_gpu')
old="    generate=runpy.run_path(str(snap/'test2026/metal/make_vector_vision_context.py'))['generate']";assert source.count(old)==1
source=source.replace(old,"    sys.path.insert(0,str(snap/'test2026/metal'))\n    sys.modules.pop('make_serial_vision_context',None)\n"+old+"\n    sys.path.pop(0)")
old="generate((snap/'src/backends/metal/metal_context.mm').read_text(),(snap/'test2026/metal/vision_fused_context_gpu.mm').read_text())";assert source.count(old)==1
source=source.replace(old,"generate((snap/'src/backends/metal/metal_context.mm').read_text(),(snap/'test2026/metal/vision_fused_context_gpu.mm').read_text(),(snap/'src/backends/metal/serial_vision_prefetch_shader.h').read_text())")
source=source.replace('list(range(5))','list(range(3))').replace('set(range(5))','set(range(3))')
old="    for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha";assert source.count(old)==1
source=source.replace(old,old+"\n    assert 'Apple M4' in meta['hardware']['machdep.cpu.brand_string'] and int(meta['hardware']['hw.memsize'])==16*2**30")
old="            assert c['passed']==(c['bits_exact'] and 0<=c['fp64_worst_ratio']<=1)";assert source.count(old)==1
source=source.replace(old,old+"\n            if variant==0:assert c['bits_exact'] and c['max_abs_error']==0")
source=source.replace('Synthetic original-order vision component only;','Synthetic vector-dimension vision component only; current scalar-prefetch8 control, same-pipeline null and unchanged softmax. Timing admission is not a speed-selection rule;')
(s/'verify_vector_vision_context.py').write_bytes(source.encode())
