"""Full nineteen-stage validation with isolated coder-head CTest and MLP off."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_joined_global_mlp_checks.py').read_text().replace('build-joined-global-mlp','build-single-coder-head').replace('METAL_JOINED_GLOBAL_MLP','METAL_SINGLE_CODER_HEAD')
exec(compile(source,str(Path(__file__)),'exec'))
