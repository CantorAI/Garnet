"""Portable Linux child wall time/RSS reporting without an extra package."""
import sys,time,resource,subprocess,json,os,signal
from pathlib import Path
max_rss=int(os.environ.get('GARNET_PI_MAX_RSS_MIB','0'))*1024
min_available=int(os.environ.get('GARNET_PI_MIN_AVAILABLE_MIB','0'))*1024
timeout=float(os.environ.get('GARNET_PI_TIMEOUT_SECONDS','0'))
if max_rss<0 or min_available<0 or timeout<0:raise ValueError('process guard limits must be nonnegative')
start=time.monotonic()
process=subprocess.Popen(sys.argv[1:],start_new_session=True)
guard_reason=None
try:
    while process.poll() is None:
        try:
            fields=dict(line.split(':',1) for line in Path('/proc/'+str(process.pid)+'/status').read_text().splitlines() if ':' in line)
            rss=int(fields.get('VmRSS','0 kB').split()[0])
            if max_rss and rss>max_rss:guard_reason='child RSS limit exceeded'
            if min_available:
                memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
                if int(memory['MemAvailable'].split()[0])<min_available:guard_reason='host memory reserve reached'
        except FileNotFoundError:pass
        if timeout and time.monotonic()-start>timeout:guard_reason='experiment time limit exceeded'
        if guard_reason:
            print('process_guard',guard_reason,flush=True)
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            break
        time.sleep(.1)
    process.wait()
finally:
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
print('process_metrics',json.dumps(dict(exit_code=process.returncode,wall_seconds=time.monotonic()-start,
    max_rss_kib=usage.ru_maxrss,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,
    major_faults=usage.ru_majflt,minor_faults=usage.ru_minflt,guard_reason=guard_reason)),flush=True)
raise SystemExit(process.returncode)
