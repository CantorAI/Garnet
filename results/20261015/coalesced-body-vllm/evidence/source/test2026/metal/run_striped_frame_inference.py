"""Actual FP32 GPU audit and sequential B1/B2 factorial frontend comparison."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('run_webp_thread_inference.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.index('destination=')],__file__,'exec'),ns)
source=ns['source'].replace('compare_webp_thread_inference.py','compare_striped_frame_inference.py').replace("['policies']['threaded']","['policies']['striped']")
source=source.replace('Refresh all six retained-prefill profiles and actual FP32 dtype audits; sequential GPU use.','Actual FP32 audit and both VL batch/frontend comparisons; sequential GPU use.')
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
