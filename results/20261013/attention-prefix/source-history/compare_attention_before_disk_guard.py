"""Native ABBA for GPU-verified prompt-slot lookup, all matched profiles."""
from pathlib import Path
import sys
source=Path(__file__).with_name('compare_joined_prefill.py').read_text()
source=source.replace('Native ABBA for native BF16 body projection; same frozen workloads and full outputs.',__doc__)
for old,new in [('build-joined-prefill','build-attention-prefix'),('logs/joined-prefill-checks-20261013','logs/attention-prefix-checks-20261013')]:
    assert source.count(old)==1;source=source.replace(old,new)
start=source.index('                expected_pairs=');end=source.index('                value.update(',start)
source=source[:start]+"                assert value['attention_prefix_enabled']==(policy=='candidate'),'Actual backend feature mismatch'\n"+source[end:]
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
