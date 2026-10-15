"""Run a command only after a successful owned background job.

Arguments: prior evidence directory, followed by the actual command.
This is a dependency gate; inference exclusivity remains the benchmark's job.
"""
import json,subprocess,sys,time
from pathlib import Path
prior=Path(sys.argv[1]);command=sys.argv[2:]
assert command
deadline=time.monotonic()+1800
while not (prior/'exit.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Owned predecessor did not finish in 30 minutes')
    pid=int((prior/'pid').read_text())
    probe=subprocess.run(['/bin/ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
    # The wrapper can publish its result and exit between the first file
    # check and ps. Prefer that authoritative terminal result to a stale PID.
    if (prior/'exit.json').exists():break
    if probe.returncode:raise RuntimeError('Owned predecessor stopped without publishing an exit result')
    identity=probe.stdout
    assert 'background_comparison.py run' in identity and str(prior) in identity
    time.sleep(5)
status=json.loads((prior/'exit.json').read_text())
assert status['exit_code']==0,status
print('Executing after successful owned predecessor',json.dumps(command),flush=True)
subprocess.run(command,check=True)
