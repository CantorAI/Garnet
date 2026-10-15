"""Fresh dtype audits and all six sequential matched profiles; preserve policy."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_retained_prefill_vllm_profiles.py').read_text()
source=source.replace('import ast,hashlib,json,os,signal,subprocess,sys,time','import ast,hashlib,json,os,signal,subprocess,sys,time,shutil')
source=source.replace('compare_retained_prefill_vllm.py','compare_global_alpha_vllm.py')
needle='py=str(r/\'comparators/vllm/venv/bin/python\')'
assert source.count(needle)==1
source=source.replace(needle,needle+'''
original_guard=guard
def guard():
    space=shutil.disk_usage(r)
    if space.free<2*2**30:raise OSError('Investigation disk free space below 2 GiB')
    original_guard()
    for line in snapshot().splitlines():
        fields=line.split()
        if fields and fields[-1].lower().endswith(('_gpu','_probe')):raise RuntimeError('Other GPU probe PID '+fields[0])
''')
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
