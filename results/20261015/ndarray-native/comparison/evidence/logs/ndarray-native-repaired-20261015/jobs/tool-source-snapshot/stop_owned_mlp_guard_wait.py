"""Interrupt only the verified owned job stalled on an obsolete service fingerprint."""
from pathlib import Path
import json,os,signal,subprocess,sys,time

root=Path(sys.argv[1]).resolve();folder=root/'logs/mlp-prefill-native-checks-20261015'
assert not (folder/'exit.json').exists()
pid=json.loads((folder/'pid.json').read_text())['pid'];assert pid==54555
args=subprocess.check_output(['/bin/ps','-p',str(pid),'-o','args='],text=True)
assert 'launch_mlp_prefill_native_checks.py' in args and str(folder) in args and '--worker' in args
assert os.getpgid(pid)==pid
diag=json.loads((folder/'guard-process-diagnostic.json').read_text())
assert {v['pid'] for v in diag['processes']}=={68650,68783}
assert all(v['lsof_exit_code']==0 and not v['inference_or_gpu_library_names'] for v in diag['processes'])
lines=subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,comm='],text=True).splitlines()
members=[]
for line in lines:
    fields=line.strip().split(None,2)
    if len(fields)==3 and int(fields[1])==pid:
        assert 'python' in fields[2].lower(), 'Owned group is no longer just the waiting Python drivers'
        members.append(int(fields[0]))
assert pid in members
log=(folder/'output.log').read_text();assert log.rstrip().endswith('Discarded interrupted validation case Qwen3.5-0.8B-private-kv 0 1')
record=dict(pid=pid,signal='SIGINT',owned_group_members=members,
            reason='WorldSense supervisor fingerprint changed; no loaded inference/GPU libraries in diagnostic. Preserve this attempt before precise guard repair.',
            stopped_at_unix=time.time(),foreign_processes_untouched=True)
(folder/'owned-stop.json').write_text(json.dumps(record,indent=2)+'\n')
os.killpg(pid,signal.SIGINT)
print(json.dumps(record))
