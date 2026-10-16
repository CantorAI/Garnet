"""Measure sequential versus chunked prefill without concurrent GPU jobs."""
import json,subprocess,sys
from pathlib import Path
runtime,build,source,suite=sys.argv[1:];suite=Path(suite)
workloads=[('image-sequential-b1','image-long:1'),('image-chunk2-b1','image-chunk:1:2'),('image-chunk4-b1','image-chunk:1:4'),('image-sequential-b2','image-long:2'),('image-chunk4-b2','image-chunk:2:4'),('text-chunk4-b1','text-chunk:1:4'),('text-chunk4-b2','text-chunk:2:4')]
for name,mode in workloads:
 folder=suite.parent/(suite.name+'-'+name);folder.mkdir(parents=True,exist_ok=True)
 if (folder/'exit.json').exists():raise RuntimeError('Use a fresh suite directory')
 print('Starting',name,flush=True)
 with (folder/'output.log').open('w') as log:
  result=subprocess.run([sys.executable,str(Path(source)/'test2026/metal/background_test.py'),'run',str(folder),runtime,build,source,mode],stdout=log,stderr=subprocess.STDOUT)
 if result.returncode or json.loads((folder/'exit.json').read_text())['exit_code']:raise RuntimeError('Benchmark failed '+name)
 print('Completed',name,flush=True)
