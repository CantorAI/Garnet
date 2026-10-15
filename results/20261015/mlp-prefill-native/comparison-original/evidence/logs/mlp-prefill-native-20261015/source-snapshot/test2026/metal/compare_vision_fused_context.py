"""Native control/candidate ABBA, fresh image encoding and complete outputs."""
from pathlib import Path
import sys
script=Path(__file__).with_name('compare_joined_prefill.py')
source=script.read_text()
source=source.replace('Native ABBA for native BF16 body projection; same frozen workloads and full outputs.',__doc__)
source=source.replace('build-joined-prefill','build-vision-fused-context').replace('logs/joined-prefill-checks-20261013','logs/vision-fused-context-checks-20261013')
start=source.index('                expected_pairs=')
end=source.index('                value.update(',start)
source=source[:start]+source[end:]
if '--model' not in sys.argv:sys.argv.extend(['--model','Qwen3.5-0.8B'])
assert sys.argv[sys.argv.index('--model')+1]=='Qwen3.5-0.8B'
destination=Path(sys.argv[2]).with_suffix('.executed.py')
destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
