"""Fresh actual FP32 audit followed by both VL batch/frontend comparisons."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_retained_prefill_vllm_profiles.py').read_text()
source=source.replace("('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B')", "('Qwen3.5-0.8B',)")
source=source.replace('compare_retained_prefill_vllm.py','compare_webp_thread_inference.py')
source=source.replace("v['summary']['decode_target_met'] and v['summary']['ttft_target_met']", "v['summary']['policies']['threaded']['decode_target_met'] and v['summary']['policies']['threaded']['ttft_target_met']")
source=source.replace('assert len(profiles)==6 and len(audits)==3','assert len(profiles)==2 and len(audits)==1')
needle="py=str(r/'comparators/vllm/venv/bin/python')"
source=source.replace(needle,needle+'''
original_guard=guard
def guard():
    import shutil
    if shutil.disk_usage(r).free<2*2**30:raise OSError('Disk free space below 2 GiB')
    original_guard()
    for line in snapshot().splitlines():
        f=line.split()
        if f and f[-1].lower().endswith(('_gpu','_probe')):raise RuntimeError('Other GPU probe PID '+f[0])
''')
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
