"""Measure actual process RSS with all three native Garnet models resident."""
import json,resource,subprocess,sys,time
from pathlib import Path
runtime,build,source=sys.argv[1:];root=Path(source).parent
for name in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
    manifest=json.loads((Path(source)/'test2026/metal/dense_models'/(name+'.json')).read_text())
    for entry in manifest['files']:
        if (root/'models'/name/entry['name']).stat().st_size!=entry['size']:raise ValueError('Pinned checkpoint size mismatch')
if (root/'models/Qwen3.5-0.8B/model.safetensors').stat().st_size!=1746942600:raise ValueError('Qwen3.5 checkpoint size mismatch')
start=time.monotonic();result=subprocess.run([runtime,str(Path(source)/'test2026/metal/run_three_resident.py'),build,source,str(root)])
print('metal_three_resident_process',json.dumps(dict(exit_code=result.returncode,wall_seconds=time.monotonic()-start,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,memory_scope='peak macOS process RSS bytes, all three loaded')),flush=True)
raise SystemExit(result.returncode)
