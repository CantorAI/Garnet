"""Explicit FP32 FMA candidate; mandatory original exact bits, bounds and safety."""
from pathlib import Path

p=Path(__file__).with_name('run_coalesced_eight_body.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
source=ns['source']
old='from make_coalesced_eight_body import context as generate_context,header as generate_header,harness as generate_harness'
assert source.count(old)==1
source=source.replace(old,'from make_eight_fma_body import context as generate_context,header as generate_header,harness as generate_harness,shader as generate_shader')
old="s/'make_coalesced_eight_body.py'";assert source.count(old)==1
source=source.replace(old,"s/'make_eight_fma_body.py',"+old)
old="(out/'coalesced_eight_body_shader.h').write_bytes((s/'coalesced_eight_body_shader.h').read_bytes())"
assert source.count(old)==1
source=source.replace(old,"(out/'coalesced_eight_fma_shader.h').write_bytes(generate_shader((s/'coalesced_eight_body_shader.h').read_text()).encode())")
old='Private current compact four-output decode versus eight-output reuse, unchanged exact BF16 payload and FP32 accumulation order.'
assert source.count(old)==1
source=source.replace(old,'Private current compact four-output decode versus lexical eight-output explicit FP32 FMA accumulation. Original weight bits and original exact-output/numerical gates remain mandatory.')
source=repr(__doc__)+source[source.index('\n'):]
exec(compile(source,__file__,'exec'))
