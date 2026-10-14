"""Recompute the original exact-component and unchanged FP64/control gates."""
from pathlib import Path
source=Path(__file__).with_name('verify_joined_activation_decode_probe.py').read_text()
start=source.index("assert proof['jobs']['joined-activation-decode-probe")
end=source.index("meta=json.loads",start)
source=source[:start]+"assert proof['jobs']['coalesced-activation-decode-20261015']['exit_code']==0\njobs=root/'evidence/logs/coalesced-activation-decode-20261015/jobs'\n"+source[end:]
source=source.replace('joined-activation-decode-repair-probe-20261014','coalesced-activation-decode-20261015').replace('joined_activation_decode_gpu','coalesced_activation_decode_gpu')
exec(compile(source,__file__,'exec'))
