"""Durable quiet-gated fresh build and full nineteen-stage head validation."""
from pathlib import Path
source=Path(__file__).with_name('launch_single_coder_head_checks.py').read_text().replace('run_guarded_single_coder_head_checks.py','run_guarded_single_coder_head_repair_checks.py')
exec(compile(source,str(Path(__file__)),'exec'))
