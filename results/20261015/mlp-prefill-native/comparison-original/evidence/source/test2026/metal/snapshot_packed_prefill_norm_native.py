"""Bounded historical snapshot; verify both actual owned handles and libraries."""
from pathlib import Path
source=Path(__file__).with_name('snapshot_residual_norm_native.py').read_text()
source=source.replace('residual-norm-checks-20261014','packed-prefill-norm-checks-20261014').replace('residual-norm-native-20261014','packed-prefill-norm-native-20261014')
source=source.replace('35350','68972').replace('36987','70597').replace('launch_residual_norm_checks.py','launch_quiet_packed_prefill_norm_checks.py').replace('queue_residual_norm_profiles.py','queue_packed_prefill_norm_profiles.py')
needle="('packed-prefill-norm-native-20261014',70597,'queue_packed_prefill_norm_profiles.py')"
assert source.count(needle)==1
source=source.replace(needle,needle+",('packed-prefill-norm-resume-checks-20261014',75785,'launch_resume_packed_prefill_norm_checks.py'),('packed-prefill-norm-resume-native-20261014',78978,'queue_resume_packed_prefill_norm_profiles.py')")
source=source.replace("str(folder),'--worker']","str(folder),*(['checks'] if script=='launch_quiet_packed_prefill_norm_checks.py' else []),'--worker']")
source=source.replace('build-residual-norm','build-packed-prefill-norm').replace('residual-norm-staging','packed-prefill-norm-staging').replace('install_residual_norm_sources.py','install_packed_prefill_norm_sources.py')
exec(compile(source,str(Path(__file__)),'exec'))
