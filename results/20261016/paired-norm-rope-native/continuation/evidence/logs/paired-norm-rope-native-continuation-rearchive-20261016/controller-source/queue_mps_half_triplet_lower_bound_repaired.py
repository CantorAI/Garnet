"""Same single synthetic probe; resume exact dependency chain after archived upstream failure."""
from pathlib import Path
p=Path(__file__).with_name('queue_mps_half_triplet_lower_bound.py');source=p.read_text()
source=source.replace('queue_mps_half_triplet_lower_bound.py','queue_mps_half_triplet_lower_bound_repaired.py')
source=source.replace('launch_mps_half_triplet_lower_bound.py','launch_mps_half_triplet_lower_bound_repaired.py')
source=source.replace('logs/compact-fused-projection-groups-queue-20261016','logs/compact-fused-projection-groups-queue-repaired-20261016')
source=source.replace('queue_compact_fused_projection_groups.py','queue_compact_fused_projection_groups_repaired.py')
exec(compile(source,__file__,'exec'))
