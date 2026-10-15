"""Run native workload benchmarks sequentially, with durable per-job evidence."""
import json,subprocess,sys
from pathlib import Path
runtime,build,source,suite=sys.argv[1:]
suite=Path(suite)
for name,mode in [('image-b1','image-long:1'),('image-b2','image:2'),('text-b1','text-long:1'),('text-b2','text-long:2')]:
 folder=suite.parent/(suite.name+'-'+name);folder.mkdir(parents=True,exist_ok=True)
 if (folder/'exit.json').exists():raise RuntimeError('Use a fresh suite directory to preserve prior evidence')
 print('Starting benchmark',name,flush=True)
 with (folder/'output.log').open('w') as log:
  result=subprocess.run([sys.executable,str(Path(source)/'test2026/metal/background_test.py'),'run',str(folder),runtime,build,source,mode],stdout=log,stderr=subprocess.STDOUT)
 proof=json.loads((folder/'exit.json').read_text())
 if result.returncode or proof['exit_code']:raise RuntimeError('Failed benchmark '+name)
 print('Completed benchmark',name,flush=True)
