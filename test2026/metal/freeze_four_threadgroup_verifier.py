"""Materialize the original audited assertions, eliminating live wrapper dependencies."""
from pathlib import Path
import ast,hashlib
p=Path(__file__).with_name('verify_four_threadgroup_probe.py');wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
expanded=ns['source']
marker="exec(compile(prefix,__file__,'exec'))";assert expanded.count(marker)==1
before,after=expanded.split(marker)
ns={'__file__':str(p)};exec(compile(before,str(p),'exec'),ns)
source='''"""Frozen exact/numerical/safety/packing/timing assertions for same-pipeline geometry."""
'''+ns['prefix']+'\n'+after
ast.parse(source)
target=p.with_name('verify_four_threadgroup_probe_frozen.py');target.write_bytes(source.encode())
print(target.name,hashlib.sha256(target.read_bytes()).hexdigest())
