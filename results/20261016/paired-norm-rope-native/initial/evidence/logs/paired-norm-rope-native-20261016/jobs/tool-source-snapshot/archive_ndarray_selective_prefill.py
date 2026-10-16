"""Freeze terminal selective NDArray component, unchanged assets and identities."""
from pathlib import Path
import sys
assert not Path(sys.argv[2]).exists()
p=Path(__file__).with_name('archive_mpsgraph_prefill.py');source=p.read_text()
source=source.replace("['mpsgraph-prefill-20261015']", "['ndarray-selective-prefill-20261016']")
source=source.replace('build-interleaved-bf16-body','build-coalesced-bf16-body')
source=source.replace("'archive_mpsgraph_prefill.py','launch_mpsgraph_prefill.py'", "'archive_ndarray_selective_prefill.py','launch_ndarray_selective_prefill.py','archive_mpsgraph_prefill.py'")
exec(compile(source,__file__,'exec'))
