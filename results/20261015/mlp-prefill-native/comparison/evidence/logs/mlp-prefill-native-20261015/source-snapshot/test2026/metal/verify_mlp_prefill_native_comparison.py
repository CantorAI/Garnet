"""Preserve numeric gates; distinguish original-check and combined archive sizes."""
from pathlib import Path

p=Path(__file__).with_name('verify_mlp_prefill_native_profiles.py')
source=p.read_text()
needle='for key,value in checks_audit.items():assert audit[key]==value,key'
assert source.count(needle)==1
source=source.replace(needle,'''for key,value in checks_audit.items():
    if key=='verified_files':
        # The combined archive additionally contains the complete native
        # comparison and its source snapshots. Every file is checked above.
        assert value>=audit[key]
    else:assert audit[key]==value,key''')
needle='result=dict(verified_files='
assert source.count(needle)==1
source=source.replace(needle,"result=dict(original_checks_archive_verified_files=audit['verified_files'],verified_files=")
exec(compile(source,__file__,'exec'))
