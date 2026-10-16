"""Preserve terminal projection/argmax component evidence, including failures."""
from pathlib import Path
source=Path(__file__).with_name('archive_joined_activation_decode_probe.py').read_text()
source=source.replace("glob('joined-activation-decode-*20261014')","glob('fused-head-argmax-*20261015')")
source=source.replace('joined_activation_decode_gpu','fused_head_argmax_gpu')
exec(compile(source,__file__,'exec'))
