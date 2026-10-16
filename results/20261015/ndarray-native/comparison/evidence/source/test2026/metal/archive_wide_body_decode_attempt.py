"""Preserve failed wide-body compilation/runtime attempts before source repair."""
from pathlib import Path
source=Path(__file__).with_name('archive_mpsgraph_prefill.py').read_text()
source=source.replace("['mpsgraph-prefill-20261015']","['wide-body-decode-20261015']")
source=source.replace("'archive_mpsgraph_prefill.py','launch_mpsgraph_prefill.py'","'archive_wide_body_decode_attempt.py','launch_wide_body_decode.py'")
exec(compile(source,__file__,'exec'))
