"""Require quiet resources before full row-specialized coder-head validation."""
from pathlib import Path
source=Path(__file__).with_name('launch_joined_global_mlp_checks.py').read_text().replace('run_guarded_joined_global_mlp_checks.py','run_guarded_single_coder_head_checks.py')
exec(compile(source,str(Path(__file__)),'exec'))
