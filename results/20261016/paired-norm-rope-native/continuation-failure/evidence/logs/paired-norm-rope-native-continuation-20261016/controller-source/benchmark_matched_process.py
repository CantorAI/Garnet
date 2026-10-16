"""One child invocation with independent process RSS/exit accounting on macOS."""
import json,resource,subprocess,sys,time
start=time.monotonic();command=sys.argv[1:]
result=subprocess.run(command)
print('matched_process',json.dumps(dict(command=command,exit_code=result.returncode,
    wall_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    memory_scope='RUSAGE_CHILDREN high-water individual child RSS, macOS bytes; not aggregate vLLM worker RSS or unique GPU allocation')),flush=True)
raise SystemExit(result.returncode)
