"""Temporary intermediate-output diagnostics for the native vision encoder."""
import os,subprocess,sys
from pathlib import Path
for stage in ('QKV','ROPE','PROB','CONTEXT','ATTN'):
    print('vision diagnostic stage',stage,flush=True)
    environment=os.environ.copy();environment['GARNET_METAL_DEBUG_VISION_'+stage]='1'
    subprocess.run([sys.executable,str(Path(__file__).with_name('test_vision_graph.py'))]+sys.argv[1:],env=environment,check=True)
