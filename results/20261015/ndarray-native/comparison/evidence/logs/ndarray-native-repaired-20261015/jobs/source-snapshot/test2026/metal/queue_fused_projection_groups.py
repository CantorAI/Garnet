"""Wait for the exact owned frontend inference job before compiling/timing GPU work."""
from pathlib import Path
source=Path(__file__).with_name('queue_threaded_striped_probe.py').read_text()
source=source.replace('striped-frame-inference-confirm-20261015','threaded-striped-inference-20261015').replace('pid=30402','pid=43673').replace('launch_confirm_striped_frame_inference.py','launch_threaded_striped_inference.py').replace('run_threaded_striped_probe.py','run_fused_projection_groups.py')
exec(compile(source,__file__,'exec'))
