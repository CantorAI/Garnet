"""Reuse all completed 19 gates; repair only XLang reference-loop compatibility."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();failed=root/'logs/fused-head-argmax-native-20261015'
assert json.loads((failed/'exit.json').read_text())['exit_code']==1
log=(failed/'recovery/Qwen3-0.6B-session.log').read_text()
assert 'NameError' in log and "name 'row' is not defined" in log
archive=root/'logs/fused-head-argmax-initial-failure-evidence.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='d1c7c7cea9681d70832c29205f3ca8bbd69eb24a8cc3557dcf3184880b6c8712'
assert hashlib.sha256((root/'build-fused-head-argmax/libgarnet.dylib').read_bytes()).hexdigest()=='8cfcfb392d19fad68695ad7d80c8c3b5989f01ca956fa50370c7b2f960230177'
source=Path(__file__).with_name('queue_fused_head_argmax_profiles.py').read_text()
source=source.replace("helpers=('queue_fused_head_argmax_profiles.py'", "helpers=('launch_repaired_fused_head_argmax_profiles.py','queue_fused_head_argmax_profiles.py'")
exec(compile(source,__file__,'exec'))
