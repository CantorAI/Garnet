import datetime,json,subprocess
from pathlib import Path
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/native-mtp3-shared-joined-intentional-stop.json'
assert not out.exists()
ledger=r/'results/native-mtp3-shared-joined-workflow-ledger/commands.jsonl'
events=[json.loads(line) for line in ledger.read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==0,'Wait for successful pilot completion'
paths=list((r/'results/native-mtp3-shared-joined-profile-pilot/benchmark').glob('*-manifest.json'));assert len(paths)==1
assert json.loads(paths[0].read_text())['status']=='completed'
raw=subprocess.check_output(['powershell','-NoProfile','-Command','Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress'])
processes=json.loads(raw)
matches=[p for p in processes if p['Name'].lower()=='python.exe' and 'native_http_bridge.py' in (p['CommandLine'] or '') and 'native-mtp3-shared-joined-worker' in p['CommandLine'] and 'run_logged.py' not in p['CommandLine']]
assert len(matches)==1,'Ambiguous study server identity'
bridge=matches[0];assert '5090-study-20261023' in bridge['CommandLine']
result=subprocess.run(['taskkill','/PID',str(bridge['ProcessId']),'/T','/F'],capture_output=True,text=True)
out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Intentional study shutdown after terminal successful pilot; preserve historical gate rejection separately','bridge':bridge,'taskkill_exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr},indent=2))
assert result.returncode==0
print('IDENTITY_GUARDED_SHARED_STUDY_SERVER_STOPPED')
