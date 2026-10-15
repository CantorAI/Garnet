"""Actual GPU precision audit followed by sequential VL B1/B2 comparisons."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('run_striped_frame_inference.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns)
source=ns['source'].replace('compare_striped_frame_inference.py','compare_threaded_striped_inference.py').replace("['policies']['striped']","['policies']['combined']")
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
