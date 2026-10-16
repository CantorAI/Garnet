"""Stop only the verified NDArray driver group between self-guarded primitive retries."""
from pathlib import Path
import json,os,signal,subprocess,sys,time
root=Path(sys.argv[1]).resolve();folder=root/'logs/ndarray-native-checks-20261015'
pid=json.loads((folder/'pid.json').read_text())['pid'];assert pid==81353
if (folder/'exit.json').exists():
    print(json.dumps(dict(already_terminal=json.loads((folder/'exit.json').read_text()))));sys.exit(0)
args=subprocess.check_output(['/bin/ps','-p',str(pid),'-o','args='],text=True)
assert 'launch_ndarray_native_checks.py' in args and str(folder) in args and '--worker' in args
assert os.getpgid(pid)==pid
for _ in range(100):
    rows=[]
    for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,comm='],text=True).splitlines():
        fields=line.strip().split(None,2)
        if len(fields)==3:rows.append((int(fields[0]),int(fields[1]),fields[2]))
    if not any(name==str(folder/'jobs/ndarray_prefill_cache_gpu') for _,_,name in rows):break
    time.sleep(.1)
else:raise RuntimeError('Owned primitive still executing; leave it untouched')
members=[p for p,g,name in rows if g==pid]
assert pid in members and all('python' in name.lower() for p,g,name in rows if g==pid)
record=dict(pid=pid,signal='SIGINT',owned_python_group_members=members,reason='Guard misclassifies own exact NDArray cache primitive; archive before scoped repair.',foreign_processes_untouched=True,at_unix=time.time())
(folder/'owned-stop.json').write_text(json.dumps(record,indent=2)+'\n');os.killpg(pid,signal.SIGINT);print(json.dumps(record))
