"""Freeze bounded current status after verifying exact owned job handles."""
from pathlib import Path
source=Path(__file__).with_name('snapshot_residual_norm_native.py').read_text()
source=source.replace('residual-norm-checks-20261014','vision-global-alpha-checks-20261014').replace('residual-norm-native-20261014','vision-global-alpha-native-20261014')
source=source.replace('35350','13040').replace('36987','16264').replace('launch_residual_norm_checks.py','launch_vision_global_alpha_checks.py').replace('queue_residual_norm_profiles.py','queue_vision_global_alpha_profiles.py')
source=source.replace("str(folder),'--worker']","str(folder),*(['checks'] if script=='launch_vision_global_alpha_checks.py' else []),'--worker']")
source=source.replace('build-residual-norm','build-vision-global-alpha').replace('residual-norm-staging','vision-global-alpha-staging').replace('install_residual_norm_sources.py','install_vision_global_alpha_sources.py')
source=source.replace('for name,expected in [', "for name,expected in [('build-global-half-alpha','887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'),")
exec(compile(source,str(Path(__file__)),'exec'))
