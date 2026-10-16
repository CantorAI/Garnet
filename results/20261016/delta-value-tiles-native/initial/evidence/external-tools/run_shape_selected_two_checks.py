"""All original28 gates; coalesced primitive additionally covers selected shapes."""
from pathlib import Path
import sys
p=Path(__file__).with_name('run_guarded_coalesced_body_native_checks.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
source=ns['source'].replace('build-coalesced-bf16-body','build-shape-selected-two-body')
assert 'assert len(statuses)==28' in source
destination=Path(sys.argv[2]).with_suffix('.complete28.executed.py');destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
