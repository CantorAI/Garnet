"""Wait for the existing worker, then strict head-reuse gates and one matched pilot."""
import json,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen

def main():
    root=Path(sys.argv[1]);source=root/'Garnet/test2026/cuda/qwen38_5090'
    sys.path.insert(0,str(source))
    from validate_native_mtp_http import validate_worker_profile
    out=root/'results/native-mtp3-shared-headreuse-workflow';out.mkdir(exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Preserve previous workflow attempt')
    diagnostic=root/'results/shared-head-reuse-http-diagnostic';pilot=root/'results/native-mtp3-shared-headreuse-profile-pilot'
    if any(p.exists() and any(p.iterdir()) for p in (diagnostic,pilot)):raise FileExistsError('Preserve previous gate/pilot artifacts')
    deadline=time.monotonic()+1800
    print('WAITING_FOR_ORIGINAL_HEAD_REUSE_WORKER; no restart on observation timeout',flush=True)
    while True:
        try:
            with urlopen('http://127.0.0.1:18082/health',timeout=5) as response:health=json.load(response)
        except Exception as error:
            if time.monotonic()>=deadline:raise TimeoutError('Inspect the original server handle; do not restart solely on this deadline') from error
            time.sleep(5);continue
        validate_worker_profile(health,3,4,'direct',4,128,True,True);break
    (out/'ready-health.json').write_text(json.dumps(health,indent=2))
    subprocess.run([sys.executable,str(source/'probe_shared_mtp_sequences.py'),str(root),str(diagnostic),'--reuse-verified-head'],check=True)
    subprocess.run([sys.executable,str(source/'run_shared_mtp_candidate.py'),str(root),str(pilot),'--diagnostic',str(diagnostic),'--reuse-verified-head'],check=True)
    (out/'completed.json').write_text(json.dumps({'status':'completed','diagnostic':str(diagnostic),'pilot':str(pilot),
        'scope':'Strict head-profile readiness, twenty self-target pairs, pinned same-profile goldens, exercised head counters and real SSE before B1 short-prompt pilot. Not broad quality, tuned references, long-context/concurrency or 2x qualification.'},indent=2))

if __name__=='__main__':main()
