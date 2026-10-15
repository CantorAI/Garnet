"""Preserve terminal fused-grid trials, generated source and exact executed scripts."""
from pathlib import Path
source=Path(__file__).with_name('archive_joined_activation_decode_probe.py').read_text()
source=source.replace("glob('joined-activation-decode-*20261014')","glob('fused-projection-groups-*20261015')").replace('joined_activation_decode_gpu','fused_projection_group_gpu')
exec(compile(source,__file__,'exec'))
