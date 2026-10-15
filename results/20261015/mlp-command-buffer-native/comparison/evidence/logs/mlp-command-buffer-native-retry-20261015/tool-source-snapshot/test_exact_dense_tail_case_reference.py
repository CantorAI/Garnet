"""One original dense-tail fixture; advance RNG identically over skipped cases."""
from pathlib import Path
import sys
selected=int(sys.argv.pop());assert selected in range(8)
source=Path(__file__).with_name('test_exact_dense_tail_reference.py').read_text()
source=source.replace('rng=np.random.default_rng(765);capacity=128','rng=np.random.default_rng(765);capacity=128;case_index=-1')
needle='  for chunk in (22,26):\n'
assert source.count(needle)==1
source=source.replace(needle,needle+'''   case_index+=1
   if case_index!=selected:
    for _ in range(3):rng.normal(size=(batch,chunk,c['hidden_size']))
    continue
''')
exec(compile(source,str(Path(__file__)),'exec'))
print('Original dense-tail fixture passed case_index='+str(selected),flush=True)
