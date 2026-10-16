"""Same component design; follow the audited native confirmation controller after filename repair."""
from pathlib import Path
p=Path(__file__).with_name('launch_compact_fused_projection_groups.py');source=p.read_text()
source=source.replace('logs/vector-vision-native-continuation-20261016','logs/vector-vision-native-continuation-repaired-20261016')
exec(compile(source,__file__,'exec'))
