"""Wait for exact owned INT8 confirmation before any packed-index GPU work."""
from pathlib import Path
source=Path(__file__).with_name('queue_threaded_striped_probe.py').read_text()
source=source.replace('striped-frame-inference-confirm-20261015','screened-head-confirm-20261015').replace('pid=30402','pid=84995').replace('launch_confirm_striped_frame_inference.py','launch_confirm_screened_head.py').replace('run_threaded_striped_probe.py','run_screened_head_int4.py')
exec(compile(source,__file__,'exec'))
