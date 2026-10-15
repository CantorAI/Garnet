"""All 19 original validation stages, plus the native joined-MLP CTest."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_global_half_alpha_checks.py').read_text()
needle="destination=Path(sys.argv[2]).with_suffix"
assert source.count(needle)==1
source=source.replace(needle,"source=source.replace('build-global-half-alpha','build-joined-global-mlp').replace(\"'METAL_GLOBAL_HALF_ALPHA'\",\"'METAL_GLOBAL_HALF_ALPHA','METAL_JOINED_GLOBAL_MLP'\")\n"+needle)
exec(compile(source,str(Path(__file__)),'exec'))
