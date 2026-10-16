"""Launch sequential native inference only after frozen fresh28 audit success."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
audit=json.loads((root/'shape-selected-two-checks-verification.json').read_text())
assert audit['validation_stages']==28 and audit['ctests']==7 and audit['coalesced_primitive_cases']==104
with (root/'shape-selected-two-checks-20261015.tar.gz').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()==audit['archive_sha256']
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('profiles',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_shape_selected_two_profiles.py'")
source=source.replace("str(root/'source/test2026/metal'/runner)","str(Path(__file__).with_name(runner))")
exec(compile(source,__file__,'exec'))
