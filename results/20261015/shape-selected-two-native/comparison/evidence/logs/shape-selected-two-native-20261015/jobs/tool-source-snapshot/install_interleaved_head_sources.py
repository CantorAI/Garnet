"""Install only after both layout trials and the alternative MPS trial are archived."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('interleaved-screen-index-20261015','interleaved-screen-index-confirm-20261015','mps-screen-index-20261015'):assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/interleaved-screen-index-verification.json').read_text());assert audit['successful_trials']==2 and audit['query_checks']==102 and audit['cpu_index_roundtrip_checks']==6
source=Path(__file__).with_name('install_serial_vision_sources.py').read_text()
for old,new in [('serial-vision-staging','interleaved-head-staging'),('serial-vision-context-20261015','interleaved-screen-index-20261015'),('serial-vision-context-confirm-20261015','interleaved-screen-index-confirm-20261015'),('serial-vision-original-source-20261015','interleaved-head-original-source-20261015'),('pinned serial vision files','pinned interleaved-head files')]:source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
