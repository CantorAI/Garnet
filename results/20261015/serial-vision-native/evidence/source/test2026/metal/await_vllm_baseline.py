"""Continue the owned setup job into import checks and exclusive benchmarks.

Arguments: setup evidence, xlang3, native build, source, comparator Python,
benchmark evidence. Waiting happens in a durable local job, not in SSH.
"""
import json,os,subprocess,sys,time
from pathlib import Path

setup,runtime,build,source,python,evidence=sys.argv[1:7]
projection=sys.argv[7] if len(sys.argv)>7 else 'shared'
setup=Path(setup);scripts=Path(source)/'test2026/metal';deadline=time.monotonic()+1800
print('Waiting for owned comparator setup',str(setup),flush=True)
while not (setup/'exit.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Comparator setup wait exceeded 30 minutes')
    pid=int((setup/'pid').read_text())
    command=subprocess.check_output(['/bin/ps','-p',str(pid),'-o','args='],text=True)
    assert 'background_comparison.py run' in command and str(setup) in command
    time.sleep(5)
status=json.loads((setup/'exit.json').read_text())
assert status['exit_code']==0,status
probe=[python,'-c','import vllm,vllm_metal,torch; import mlx.core as mx; assert mx.metal.is_available(); print("Comparator imports passed; Metal device",mx.metal.device_info()["device_name"])']
print('Running comparator import/device check',flush=True)
subprocess.run(probe,check=True)
command=[sys.executable,str(scripts/'compare_vllm_metal.py'),runtime,build,source,python,evidence,projection]
print('Running matched exclusive baseline',json.dumps(command),flush=True)
subprocess.run(command,check=True)
subprocess.run([sys.executable,str(scripts/'summarize_vllm_comparison.py'),evidence],check=True)
