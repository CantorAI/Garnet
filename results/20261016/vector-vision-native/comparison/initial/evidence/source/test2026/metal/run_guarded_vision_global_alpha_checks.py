"""Full nineteen-stage validation with vision-only alpha and other trials off."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_joined_global_mlp_checks.py').read_text().replace('build-joined-global-mlp','build-vision-global-alpha').replace('METAL_JOINED_GLOBAL_MLP','METAL_VISION_GLOBAL_ALPHA')
exec(compile(source,str(Path(__file__)),'exec'))
