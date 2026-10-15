"""Verify actual owned handles and freeze bounded global-alpha status evidence."""
from pathlib import Path
source=Path(__file__).with_name('snapshot_residual_norm_native.py').read_text()
source=source.replace('residual-norm-checks-20261014','global-half-alpha-checks-20261014').replace('residual-norm-native-20261014','global-half-alpha-native-20261014')
source=source.replace('35350','12334').replace('36987','15843').replace('launch_residual_norm_checks.py','launch_global_half_alpha_checks.py').replace('queue_residual_norm_profiles.py','queue_global_half_alpha_profiles.py')
source=source.replace("str(folder),'--worker']","str(folder),*(['checks'] if script=='launch_global_half_alpha_checks.py' else []),'--worker']")
source=source.replace('build-residual-norm','build-global-half-alpha').replace('residual-norm-staging','global-half-alpha-staging').replace('install_residual_norm_sources.py','install_global_half_alpha_sources.py')
exec(compile(source,str(Path(__file__)),'exec'))
