"""Preserve every terminal independent-projection trial and exact source."""
from pathlib import Path
source=Path(__file__).with_name('archive_joined_activation_decode_probe.py').read_text()
source=source.replace("glob('joined-activation-decode-*20261014')","glob('concurrent-bf16-groups-*20261015')").replace('joined_activation_decode_gpu','concurrent_bf16_group_gpu')
exec(compile(source,__file__,'exec'))
