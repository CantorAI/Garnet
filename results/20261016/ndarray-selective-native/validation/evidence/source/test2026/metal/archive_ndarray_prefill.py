"""Archive native NDArray context and independent common-half admission proof."""
from pathlib import Path
p=Path(__file__).with_name('archive_mpsgraph_prefill.py');source=p.read_text()
source=source.replace("['mpsgraph-prefill-20261015']","['ndarray-prefill-20261015']")
source=source.replace('build-interleaved-bf16-body','build-coalesced-bf16-body')
source=source.replace("'archive_mpsgraph_prefill.py','launch_mpsgraph_prefill.py'","'archive_ndarray_prefill.py','launch_ndarray_prefill.py'")
exec(compile(source,__file__,'exec'))
