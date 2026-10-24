from pathlib import Path
import json,subprocess,time
r=Path('D:/CantorAI/5090-study-20261023');ledger=r/'results/native-mtp3-shared-joined-workflow-ledger/commands.jsonl'
print('WAITING_FOR_ORIGINAL_JOINED_PILOT_BEFORE_STOP',flush=True)
while True:
 events=[json.loads(line) for line in ledger.read_text().splitlines()]
 if events and events[-1]['event']=='finished':break
 time.sleep(5)
if events[-1]['exit_code']!=0:raise RuntimeError('Pilot failed; inspect the original server and retained failure before cleanup')
subprocess.run(['C:/Python312/python.exe',str(r/'stop-shared-joined-worker.py')],check=True)
