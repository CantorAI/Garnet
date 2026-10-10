"""Portable Linux child wall time/RSS reporting without an extra package."""
import sys,time,resource,subprocess,json
start=time.monotonic()
process=subprocess.run(sys.argv[1:])
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
print('process_metrics',json.dumps(dict(exit_code=process.returncode,wall_seconds=time.monotonic()-start,
    max_rss_kib=usage.ru_maxrss,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,
    major_faults=usage.ru_majflt,minor_faults=usage.ru_minflt)),flush=True)
raise SystemExit(process.returncode)
