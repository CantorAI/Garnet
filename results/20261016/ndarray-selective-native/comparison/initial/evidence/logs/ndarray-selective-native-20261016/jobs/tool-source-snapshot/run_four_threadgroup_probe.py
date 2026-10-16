"""Per-projection same-pipeline64/128/256 scheduling, original exact gates."""
from pathlib import Path
import sys
p=Path(__file__).with_name('run_two_matrix_probe.py');wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns);source=ns['source']
old='from make_two_matrix_probe import context as generate_context,header as generate_header,harness as generate_harness,shader as generate_shader'
assert source.count(old)==1
source=source.replace(old,old.replace('make_two_matrix_probe','make_four_threadgroup_probe'))
old="s/'make_two_matrix_probe.py'";assert source.count(old)==1
source=source.replace(old,"s/'make_four_threadgroup_probe.py',s/'run_two_matrix_probe.py',"+old)
source=source.replace('metal_context_coalesced_two','metal_context_four_threadgroup').replace('two_matrix_probe_gpu','four_threadgroup_gpu')
source=source.replace('coalesced_two_body_shader.h','retained_four_shader.h')
needle='for p in paths:';at=source.index(needle)
source=source[:at]+"metadata.update(scope='Per-projection original four-output pipeline at64-thread control versus64/128/256-thread scheduling. Same shader/pipeline and original payload/math.64-thread candidate is a null comparison.',limitations='One projection per timed submission; repeated resident cache behavior differs from inference. Retained shader proof copy is not an extra compiled shader. No model or vLLM gain.',new_shader_compiled=False)\n"+source[at:]
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,__file__,'exec'))
