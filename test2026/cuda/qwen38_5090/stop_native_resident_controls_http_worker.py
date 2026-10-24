from pathlib import Path
import datetime,json,subprocess
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/native-resident-controls-http-intentional-stop.json';assert not out.exists()
def terminal(name):
 events=[json.loads(l) for l in (r/'results'/name/'commands.jsonl').read_text(encoding='utf8').splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code'] in (0,1),'Observe original handle terminal before stopping';return events[-1]['exit_code']
workflow=terminal('native-resident-controls-http-controller-ledger')
if workflow==0:
 manifests=list((r/'results/native-resident-controls-profile-pilot/benchmark').glob('*-manifest.json'));assert len(manifests)==1 and json.loads(manifests[0].read_text(encoding='utf8'))['status']=='completed'
query="@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -like '*native_http_bridge.py*' -and $_.CommandLine -like '*native-resident-controls-http-worker*' -and $_.CommandLine -notlike '*monitor_native_probe.py*' -and $_.CommandLine -notlike '*run_logged.py*'} | Select-Object ProcessId,ParentProcessId,Name,CommandLine) | ConvertTo-Json -Compress"
raw=subprocess.check_output(['powershell','-NoProfile','-Command',query]);matches=json.loads(raw or b'[]');matches=[matches] if isinstance(matches,dict) else matches
assert len(matches)==1;bridge=matches[0];assert '5090-study-20261023' in bridge['CommandLine'] and 'serve-native-resident-controls-worker.bat' in bridge['CommandLine'] and 'qwen38-27b-nvfp4' in bridge['CommandLine']
result=subprocess.run(['taskkill','/PID',str(bridge['ProcessId']),'/T','/F'],capture_output=True,text=True)
out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Intentional owned resident-controls27B study shutdown after observed terminal quality/measurement; preserve success or failure, no inferred result','workflow_exit_code':workflow,'bridge':bridge,'taskkill_exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'resource_wrapper':'Excluded so final whole-lifetime memory report can finish; observe original server handle terminal separately.'},indent=2),encoding='utf8');assert result.returncode==0
print('OWNED_RESIDENT_CONTROLS_BRIDGE_STOPPED_AFTER_TERMINAL_WORK')
