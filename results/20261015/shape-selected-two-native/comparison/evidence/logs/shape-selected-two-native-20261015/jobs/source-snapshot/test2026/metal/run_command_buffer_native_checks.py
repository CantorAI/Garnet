"""All28 original trained/native gates in a separate generated-source tree."""
from pathlib import Path
import sys

p=Path(__file__).with_name('run_guarded_mlp_prefill_native_checks.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
source=ns['source'].replace('build-mlp-prefill-graph-guardfix','build-mlp-command-buffer')
assert source.count("run('wide-batch-trained-reference',")==1
assert 'assert len(statuses)==28' in source
destination=Path(sys.argv[2]).with_suffix('.complete28.executed.py')
destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
