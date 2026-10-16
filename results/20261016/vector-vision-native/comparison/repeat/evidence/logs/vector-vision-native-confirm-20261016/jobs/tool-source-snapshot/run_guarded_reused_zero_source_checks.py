"""All original 19 gates with retained-source reset and global-alpha compute."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_global_half_alpha_checks.py').read_text();needle="destination=Path(sys.argv[2]).with_suffix";assert source.count(needle)==1
source=source.replace(needle,"source=source.replace('build-global-half-alpha','build-reused-zero-source').replace(\"'METAL_GLOBAL_HALF_ALPHA'\",\"'METAL_GLOBAL_HALF_ALPHA','METAL_REUSED_ZERO_SOURCE'\")\n"+needle)
exec(compile(source,__file__,'exec'))
