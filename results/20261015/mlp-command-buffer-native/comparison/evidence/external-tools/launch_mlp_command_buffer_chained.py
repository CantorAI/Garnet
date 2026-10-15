"""Extend the archived 64-case primitive with true two-graph dependencies."""
from pathlib import Path
import hashlib,json,sys

root=Path(sys.argv[1]).resolve()
prior=root/'logs/mlp-command-buffer-primitive-20261015'
assert json.loads((prior/'exit.json').read_text())['exit_code']==0
with (root/'mlp-command-buffer-primitive-20261015.tar.gz').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='1e87242bb5a9fc6f77ec9c6d2f73cd5dfb4236019f5ad2531557bf5e6f510a2f'
source=Path(__file__).with_name('launch_mlp_command_buffer_primitive.py').read_text()
exec(compile(source,__file__,'exec'))
