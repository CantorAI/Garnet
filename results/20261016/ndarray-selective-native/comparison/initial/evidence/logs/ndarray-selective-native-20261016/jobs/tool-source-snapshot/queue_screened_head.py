"""Wait for exact owned vision validation/performance sequence before any work."""
from pathlib import Path
source=Path(__file__).with_name('queue_threaded_striped_probe.py').read_text()
source=source.replace('striped-frame-inference-confirm-20261015','parallel-vision-native-20261015').replace('pid=30402','pid=64917').replace('launch_confirm_striped_frame_inference.py','queue_parallel_vision_profiles.py').replace('run_threaded_striped_probe.py','run_screened_head.py').replace('>1800','>7200')
exec(compile(source,__file__,'exec'))
