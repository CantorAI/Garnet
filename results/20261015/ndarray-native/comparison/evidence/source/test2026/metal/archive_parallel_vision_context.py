"""Archive all terminal parallel-vision component trials, including failures."""
from pathlib import Path
source=Path(__file__).with_name('archive_joined_activation_decode_probe.py').read_text()
source=source.replace("glob('joined-activation-decode-*20261014')","glob('parallel-vision-context-*20261015')").replace('joined_activation_decode_gpu','parallel_vision_context_gpu')
exec(compile(source,__file__,'exec'))
