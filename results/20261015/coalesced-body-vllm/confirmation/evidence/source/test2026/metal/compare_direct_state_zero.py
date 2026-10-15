"""Native ABBA direct-zero candidate versus otherwise identical attention build."""
from pathlib import Path
import sys
# Resolve the existing comparator template, then adapt the candidate policy.
template=Path(__file__).with_name('compare_joined_prefill.py').read_text()
source=template.replace('import ast,hashlib,json,os,signal,statistics,subprocess,sys,time','import ast,hashlib,json,os,signal,statistics,subprocess,sys,time,shutil')
source=source.replace('def guard(label):\n    strict_guard()', '''def guard(label):
    if shutil.disk_usage(r).free<2*2**30:raise OSError('Disk free space below 2 GiB')
    strict_guard()''')
source=source.replace('build-resident-kv-inplace','build-attention-prefix').replace('build-joined-prefill','build-direct-state-zero').replace('logs/joined-prefill-checks-20261013','logs/direct-state-zero-checks-20261013')
source=source.replace('73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2','f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088')
start=source.index('                expected_pairs=');end=source.index('                value.update(',start)
source=source[:start]+"                assert value['attention_prefix_enabled']\n                assert value['direct_state_zero_enabled']==(policy=='candidate')\n"+source[end:]
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
