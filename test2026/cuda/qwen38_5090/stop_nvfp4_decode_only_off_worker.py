from pathlib import Path
import datetime,json,subprocess
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/native-mtp3-decode-only-off-intentional-stop.json';assert not out.exists()
events=[json.loads(l) for l in (r/'results/native-mtp3-decode-only-off-workflow-ledger/commands.jsonl').read_text(encoding='utf8').splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code'] in (0,1),'Observe original workflow terminal handle before stopping'
code=events[-1]['exit_code']
if code==0:assert json.loads((r/'results/native-mtp3-decode-only-off-workflow/report.json').read_text(encoding='utf8'))['status']=='off_control_exact'
query="@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -like '*native_http_bridge.py*' -and $_.CommandLine -like '*native-mtp3-decode-only-off-worker*' -and $_.CommandLine -notlike '*monitor_native_probe.py*' -and $_.CommandLine -notlike '*run_logged.py*'} | Select-Object ProcessId,ParentProcessId,Name,CommandLine) | ConvertTo-Json -Compress"
raw=subprocess.check_output(['powershell','-NoProfile','-Command',query]);matches=json.loads(raw or b'[]');matches=[matches] if isinstance(matches,dict) else matches
assert len(matches)==1,'Ambiguous owned study bridge';bridge=matches[0];assert '5090-study-20261023' in bridge['CommandLine'] and 'serve-nvfp4-decode-only-off-worker.bat' in bridge['CommandLine']
result=subprocess.run(['taskkill','/PID',str(bridge['ProcessId']),'/T','/F'],capture_output=True,text=True)
out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Intentional owned study shutdown after observed terminal '+('qualified pilot' if code==0 else 'failed gate/workflow; no success inferred'),'workflow_exit_code':code,'bridge':bridge,'taskkill_exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'resource_wrapper':'Excluded from kill so it can retain whole-lifetime sampled VRAM report; observe original server handle terminal separately.'},indent=2),encoding='utf8');assert result.returncode==0
print('IDENTITY_GUARDED_V3_BRIDGE_TREE_STOPPED')
