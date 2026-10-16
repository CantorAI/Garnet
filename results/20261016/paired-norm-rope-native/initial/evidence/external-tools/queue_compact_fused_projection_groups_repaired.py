"""Same compact probe/selection; resume dependency tracking after preserved native audit failure."""
from pathlib import Path
p=Path(__file__).with_name('queue_compact_fused_projection_groups.py');source=p.read_text()
source=source.replace('queue_compact_fused_projection_groups.py','queue_compact_fused_projection_groups_repaired.py')
source=source.replace('launch_compact_fused_projection_groups.py','launch_compact_fused_projection_groups_repaired.py')
source=source.replace('logs/vector-vision-native-continuation-20261016','logs/vector-vision-native-continuation-repaired-20261016')
source=source.replace('continue_vector_vision_native.py','continue_vector_vision_native_repaired.py')
exec(compile(source,__file__,'exec'))
