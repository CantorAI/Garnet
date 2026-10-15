"""All six original workloads, same common frontend and exact generated outputs."""
from pathlib import Path
source=Path(__file__).with_name('run_vision_global_alpha_profiles.py').read_text().replace('compare_vision_global_alpha.py','compare_parallel_vision.py')
assert "confirm='--confirm' in sys.argv[3:]" in source
source=source.replace("confirm='--confirm' in sys.argv[3:]","confirm=False")
exec(compile(source,__file__,'exec'))
