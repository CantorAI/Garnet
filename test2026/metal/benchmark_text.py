"""Launch actual native generation and report peak child RSS on macOS."""
import json,resource,subprocess,sys,time
from pathlib import Path
runtime,build,source,model=sys.argv[1:5]
batch=sys.argv[5] if len(sys.argv)>5 else '1'
limit=sys.argv[6] if len(sys.argv)>6 else '16'
model=Path(model)
expected=1746942600
if (model/'model.safetensors').stat().st_size!=expected:
    raise ValueError('Pinned full checkpoint is incomplete')
command=[runtime,str(Path(source)/'test2026/metal/run_text_decoder.py'),build,source,str(model),limit,batch]
if len(sys.argv)>7:command.append(sys.argv[7])
if len(sys.argv)>8:command.append(sys.argv[8])
started=time.monotonic()
result=subprocess.run(command)
print('metal_text_process',json.dumps(dict(exit_code=result.returncode,wall_seconds=time.monotonic()-started,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    memory_scope='peak native runtime RSS, macOS bytes; not isolated GPU allocation')),flush=True)
raise SystemExit(result.returncode)
