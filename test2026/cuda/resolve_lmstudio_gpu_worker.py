"""Resolve a loaded LM Studio GPU worker by daemon ancestry and adapter rows.

Does not select an arbitrary largest process. Requires exactly one CUDA-adapter
process with >=2GB dedicated usage belonging to the specified daemon tree.
"""
import argparse,json,subprocess
from pathlib import Path
from capture_windows_gpu_memory import adapter,counters
parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--daemon-pid',type=int,required=True)
args=parser.parse_args();mapping=adapter(0);snapshot=counters(mapping,None)
command='Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CreationDate | ConvertTo-Json -Depth 3 -Compress'
processes=json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-Command',command],text=True))
index={int(p['ProcessId']):p for p in processes}
assert args.daemon_pid in index and 'llmster' in index[args.daemon_pid]['Name'].lower(),'Daemon PID is not the current llmster process'
def ancestors(pid):
    chain=[]
    while pid in index and pid not in [int(p['ProcessId']) for p in chain]:
        chain.append(index[pid]);pid=int(index[pid]['ParentProcessId'])
    return chain
candidates={}
for row in snapshot['adapter_rows']:
    if not row['cuda_node_selected'] or int(row['DedicatedUsage'])<2_000_000_000:continue
    pid=row['process_id'];chain=ancestors(pid)
    if any(int(p['ProcessId'])==args.daemon_pid for p in chain):candidates[pid]=chain
assert len(candidates)==1,('Expected exactly one model GPU process under this daemon',sorted(candidates))
worker=next(iter(candidates));result={'worker_process_id':worker,'daemon_process_id':args.daemon_pid,
    'ancestry':candidates[worker],'adapter':mapping,
    'worker_counter_rows':[r for r in snapshot['adapter_rows'] if r['process_id']==worker],
    'scope':'Adapter-resolved counter ownership plus verified llmster ancestry; duplicate rows retained, not aggregated.'}
Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
