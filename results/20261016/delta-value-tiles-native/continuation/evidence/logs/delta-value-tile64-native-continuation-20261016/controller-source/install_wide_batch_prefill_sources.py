"""Install only after prior jobs terminated and their independent archives passed."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('interleaved-head-native-20261015','interleaved-head-native-confirm-20261015','replanned-vl-prefill-20261015','replanned-vl-prefill-confirm-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0,name
stage=root/'wide-batch-prefill-staging'
assert json.loads((stage/'replanned-vl-prefill-prior-verification.json').read_text())['full_output_requests']==36
assert json.loads((stage/'interleaved-head-native-prior-verification.json').read_text())['verified_files']==1861
source=Path(__file__).with_name('install_serial_vision_sources.py').read_text()
source=source.replace('serial-vision-staging','wide-batch-prefill-staging').replace('serial-vision-original-source-20261015','wide-batch-prefill-original-source-20261015')
exec(compile(source,__file__,'exec'))
