"""Run an entirely fresh six-profile cohort after the archived frontend failure."""
from pathlib import Path
import hashlib,json,sys

root=Path(sys.argv[1]).resolve()
repair=json.loads((root/'frontend-path-repair.json').read_text())
with (root/'command-buffer-native-20261015.tar.gz').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()==repair['original_failure_archive_sha256']
assert repair['copied_source_hashes_unchanged'] and (root/'comparators/vllm/venv/bin/python').is_file()
assert Path(sys.argv[2]).name=='mlp-command-buffer-native-retry-20261015'
source=Path(__file__).with_name('launch_command_buffer_native_profiles.py').read_text()
exec(compile(source,__file__,'exec'))
