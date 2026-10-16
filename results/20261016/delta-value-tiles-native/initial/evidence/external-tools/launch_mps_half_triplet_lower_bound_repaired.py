"""Same synthetic precision/cost probe; require the resumed compact chain's archived success."""
from pathlib import Path
p=Path(__file__).with_name('launch_mps_half_triplet_lower_bound.py');source=p.read_text()
source=source.replace('logs/compact-fused-projection-groups-queue-20261016','logs/compact-fused-projection-groups-queue-repaired-20261016')
exec(compile(source,__file__,'exec'))
