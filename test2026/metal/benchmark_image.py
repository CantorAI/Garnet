"""Run native image inference and report macOS peak child RSS."""
import json,resource,subprocess,sys,time
from pathlib import Path
runtime,build,source,model,metadata=sys.argv[1:6]
limit=sys.argv[6] if len(sys.argv)>6 else '16'
if (Path(model)/'model.safetensors').stat().st_size!=1746942600:
    raise ValueError('Pinned full checkpoint is incomplete')
command=[runtime,str(Path(source)/'test2026/metal/run_image_decoder.py'),build,source,model,metadata,limit]
started=time.monotonic();result=subprocess.run(command)
print('metal_image_process',json.dumps(dict(exit_code=result.returncode,wall_seconds=time.monotonic()-started,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    memory_scope='peak native runtime RSS, macOS bytes; not isolated GPU allocation')),flush=True)
raise SystemExit(result.returncode)
