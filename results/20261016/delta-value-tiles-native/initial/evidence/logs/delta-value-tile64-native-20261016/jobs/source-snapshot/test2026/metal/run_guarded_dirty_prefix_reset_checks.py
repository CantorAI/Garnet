"""All original nineteen stages plus bounded-reset CTest and session recovery."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_joined_global_mlp_checks.py').read_text().replace('build-joined-global-mlp','build-dirty-prefix-reset').replace('METAL_JOINED_GLOBAL_MLP','METAL_DIRTY_PREFIX_RESET')
exec(compile(source,str(Path(__file__)),'exec'))
