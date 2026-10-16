"""Wait only on the exact owned global-alpha validation handle before benchmarks."""
from pathlib import Path
source=Path(__file__).with_name('queue_packed_prefill_norm_profiles.py').read_text()
source=source.replace('packed-prefill-norm-checks-20261014','global-half-alpha-checks-20261014').replace('68972','12334')
source=source.replace('launch_quiet_packed_prefill_norm_checks.py','launch_global_half_alpha_checks.py').replace('run_packed_prefill_norm_profiles.py','run_global_half_alpha_profiles.py')
exec(compile(source,str(Path(__file__)),'exec'))
