"""Quiet-gated durable validation for the isolated streamed-prefill build."""
from pathlib import Path
source=Path(__file__).with_name('launch_vision_global_alpha_checks.py').read_text().replace('run_guarded_vision_global_alpha_checks.py','run_guarded_prefill_stream_checks.py')
exec(compile(source,__file__,'exec'))
