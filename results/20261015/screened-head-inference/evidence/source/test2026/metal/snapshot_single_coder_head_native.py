"""Verify exact live handles and retain bounded build/CTest evidence."""
from pathlib import Path
source=Path(__file__).with_name('snapshot_residual_norm_native.py').read_text()
source=source.replace('residual-norm-checks-20261014','single-coder-head-repair-checks-20261014').replace('residual-norm-native-20261014','single-coder-head-native-20261014')
source=source.replace('35350','80953').replace('36987','86463').replace('launch_residual_norm_checks.py','launch_single_coder_head_repair_checks.py').replace('queue_residual_norm_profiles.py','queue_single_coder_head_profiles.py')
source=source.replace("str(folder),'--worker']","str(folder),*(['checks'] if script=='launch_single_coder_head_repair_checks.py' else []),'--worker']")
source=source.replace('build-residual-norm','build-single-coder-head-repaired').replace('residual-norm-staging','single-coder-head-repair-staging').replace('install_residual_norm_sources.py','install_single_coder_head_repair.py')
source=source.replace('for name,expected in [', "for name,expected in [('build-global-half-alpha','887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'),")
extra="""
failed=root/'logs/single-coder-head-checks-20261014'
status['failed_initial_attempt']=json.loads((failed/'exit.json').read_text())
assert status['failed_initial_attempt']['exit_code']==1
for relative in ('pid.json','exit.json','output.log','jobs/attempts/build/case0-attempt0/build.log'):
 p=failed/relative
 if p.is_file():files['native/pending/failed-initial/'+relative]=p
for relative in ('original.json','manifest.json','source.tar.gz'):
 files['native/pending/initial-staging/'+relative]=root/'single-coder-head-staging'/relative
"""
source=source.replace('manifest={}',extra+'\nmanifest={}')
exec(compile(source,str(Path(__file__)),'exec'))
