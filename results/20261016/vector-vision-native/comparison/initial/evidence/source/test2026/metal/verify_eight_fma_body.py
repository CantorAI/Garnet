"""Regenerate explicit-FMA shader; preserve every original numerical/bit gate."""
from pathlib import Path
p=Path(__file__).with_name('verify_coalesced_eight_body.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
source=ns['source'].replace('make_coalesced_eight_body','make_eight_fma_body')
old="assert (out/'coalesced_eight_body_shader.h').read_bytes()==(snap/'test2026/metal/coalesced_eight_body_shader.h').read_bytes()"
assert source.count(old)==1
source=source.replace(old,"assert (out/'coalesced_eight_fma_shader.h').read_bytes()==factory['shader']((snap/'test2026/metal/coalesced_eight_body_shader.h').read_text()).encode()")
old="assert (out/'coalesced_eight_body_shader.h').read_bytes()==gen['generate']((snap/'src/backends/metal/coalesced_body_block_shader.h').read_text()).encode()"
assert source.count(old)==1
source=source.replace(old,"assert (snap/'test2026/metal/coalesced_eight_body_shader.h').read_bytes()==gen['generate']((snap/'src/backends/metal/coalesced_body_block_shader.h').read_text()).encode()")
source=source.replace('Private eight-output reuse of unchanged compact four-output blocks.',
    'Private lexical eight-output explicit FP32 FMA versus original compact four-output control; original exact bits and bounds mandatory.')
exec(compile(source,__file__,'exec'))
