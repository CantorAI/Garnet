"""Preserve terminal prefill study, exact generated source and timing logs."""
from pathlib import Path
source=Path(__file__).with_name('archive_joined_activation_decode_probe.py').read_text()
source=source.replace("glob('joined-activation-decode-*20261014')","glob('coalesced-activation-prefill-*20261015')").replace('joined_activation_decode_gpu','coalesced_activation_prefill_gpu')
exec(compile(source,__file__,'exec'))
