"""Private two-output compute, unchanged compact storage, original exact gates."""
from pathlib import Path
p=Path(__file__).with_name('run_eight_fma_body.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
source=ns['source']
old='from make_eight_fma_body import context as generate_context,header as generate_header,harness as generate_harness,shader as generate_shader'
assert source.count(old)==1
source=source.replace(old,old.replace('make_eight_fma_body','make_two_body'))
old="s/'make_eight_fma_body.py'";assert source.count(old)==1
source=source.replace(old,"s/'make_two_body.py',s/'run_eight_fma_body.py',"+old)
old="(out/'coalesced_eight_fma_shader.h').write_bytes(generate_shader((s/'coalesced_eight_body_shader.h').read_text()).encode())"
assert source.count(old)==1
source=source.replace(old,"(out/'coalesced_two_body_shader.h').write_bytes(generate_shader((root/'source/src/backends/metal/coalesced_body_block_shader.h').read_text()).encode())")
source=source.replace('metal_context_coalesced_eight','metal_context_coalesced_two').replace('coalesced_eight_body_gpu','coalesced_two_body_gpu')
old='Private current compact four-output decode versus lexical eight-output explicit FP32 FMA accumulation. Original weight bits and original exact-output/numerical gates remain mandatory.'
assert source.count(old)==1
source=source.replace(old,'Private current four-output control versus two-output SIMD compute with explicit FP32 FMA over the unchanged original four-output compact payload. Exact bits and original numerical gates remain mandatory.')
source=repr(__doc__)+source[source.index('\n'):]
exec(compile(source,__file__,'exec'))
