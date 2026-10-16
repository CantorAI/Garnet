"""Back up every replaced native source after exact completed component trials."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
audit=json.loads((root/'coalesced-body-native-staging/component-verification.json').read_text())
assert audit['numeric_checks']==144 and audit['control_profiles']==378 and audit['timed_executions']==2304
source=Path(__file__).with_name('install_serial_vision_sources.py').read_text()
for old,new in [('serial-vision-staging','coalesced-body-native-staging'),('serial-vision-original-source-20261015','coalesced-body-native-original-source-20261015'),('serial-vision-context-20261015','coalesced-body-block-20261015'),('serial-vision-context-confirm-20261015','coalesced-body-block-confirm-20261015')]:source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
