"""Regenerate two-output shader and audit all original exact/bound/safety gates."""
from pathlib import Path
p=Path(__file__).with_name('verify_coalesced_eight_body.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
source=ns['source']
for old,new in (('make_coalesced_eight_body','make_two_body'),('metal_context_coalesced_eight','metal_context_coalesced_two'),('coalesced_eight_body_gpu','coalesced_two_body_gpu')):source=source.replace(old,new)
old="assert (out/'coalesced_eight_body_shader.h').read_bytes()==(snap/'test2026/metal/coalesced_eight_body_shader.h').read_bytes()"
assert source.count(old)==1
source=source.replace(old,"assert (out/'coalesced_two_body_shader.h').read_bytes()==factory['shader']((snap/'src/backends/metal/coalesced_body_block_shader.h').read_text()).encode()")
old="""    unrolled=snap/'test2026/metal/make_unrolled_coalesced_eight_shader.py'
    if unrolled.exists():
        gen=runpy.run_path(str(unrolled))
        assert (out/'coalesced_eight_body_shader.h').read_bytes()==gen['generate']((snap/'src/backends/metal/coalesced_body_block_shader.h').read_text()).encode()
"""
assert source.count(old)==1
source=source.replace(old,'')
source=source.replace('Private eight-output reuse of unchanged compact four-output blocks.',
                      'Private two-output SIMD compute with explicit FP32 FMA over the unchanged four-output compact payload.')
exec(compile(source,__file__,'exec'))
