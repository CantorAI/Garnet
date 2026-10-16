"""Exclusive original BF16 kernel with row-two work made compile-time dead."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_constant_bf16_probe.py').read_text()
source=source.replace('constant_bf16_gpu','single_row_bf16_gpu').replace('make_constant_bf16_probe.py','make_single_row_bf16_probe.py')
source=source.replace("s/'make_single_row_bf16_probe.py',Path(__file__)","s/'make_single_row_bf16_probe.py',s/'make_constant_bf16_probe.py',s/'run_constant_bf16_probe.py',Path(__file__)")
start=source.index('shader_text=shader.read_text()');end=source.index("probe=out/",start)
source=source[:start]+'''shader_text=shader.read_text()
single=shader_text.replace('kernel void bf16_decode(', 'kernel void single_bf16_decode(').replace('kernel void bf16_decode_small(', 'kernel void single_bf16_decode_small(').replace('p.rows==2','false')
single=single.replace('#include <metal_stdlib>','').replace('using namespace metal;','').replace('struct BF16Params {ulong rows,inner,columns,aStride,oStride;};','')
shader.write_bytes((shader_text+single).encode());metadata['shader_sha256']=digest(shader);(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\\n')
'''+source[end:]
source=source.replace('"""Exclusive original BF16 shader weight-address-space gate with verified checkpoints."""',repr(__doc__))
source=source.replace('All constant BF16 projections passed','All row-specialized BF16 projections passed')
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
