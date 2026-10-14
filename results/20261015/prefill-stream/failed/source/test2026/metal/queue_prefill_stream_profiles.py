"""Verify the exact live streamed-prefill validation handle before native ABBA."""
from pathlib import Path
source=Path(__file__).with_name('queue_vision_global_alpha_profiles.py').read_text().replace('vision-global-alpha-checks-20261014','prefill-stream-checks-20261015').replace('launch_vision_global_alpha_checks.py','launch_prefill_stream_checks.py').replace('run_vision_global_alpha_profiles.py','run_prefill_stream_profiles.py').replace('assert pid==13040','assert pid==98669')
exec(compile(source,__file__,'exec'))
