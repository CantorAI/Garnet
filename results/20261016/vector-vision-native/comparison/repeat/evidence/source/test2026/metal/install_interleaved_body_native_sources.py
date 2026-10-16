"""Preserve all replaced source bytes only after frozen comparator jobs finish."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for name in ('wide-batch-vllm-20261015','wide-batch-vllm-confirm-20261015','interleaved-bf16-body-20261015','interleaved-bf16-body-confirm-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
for name in ('wide-batch-vllm-verification.json','wide-batch-vllm-confirm-verification.json'):
    audit=json.loads((root/'work'/name).read_text());assert audit['complete_matched_outputs']==72 and audit['actual_dtype_audits']==3
audit=json.loads((root/'interleaved-body-native-staging/component-verification.json').read_text());assert audit['numeric_checks']==144 and audit['timed_executions']==2304
source=Path(__file__).with_name('install_serial_vision_sources.py').read_text().replace('serial-vision-staging','interleaved-body-native-staging').replace('serial-vision-original-source-20261015','interleaved-body-native-original-source-20261015')
exec(compile(source,__file__,'exec'))
