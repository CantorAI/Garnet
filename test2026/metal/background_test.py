"""Keep test logs and completion evidence across intermittent SSH sessions."""
import json,os,subprocess,sys
from pathlib import Path
mode,folder,runtime,build,source,batch=sys.argv[1:]
folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
pidfile=folder/'pid';exitfile=folder/'exit.json';logfile=folder/'output.log'
if mode=='launch':
 if pidfile.exists() and not exitfile.exists():
  try:os.kill(int(pidfile.read_text()),0)
  except ProcessLookupError:pass
  else:raise RuntimeError('Prior job PID is still alive; inspect it before starting another')
 exitfile.unlink(missing_ok=True)
 with logfile.open('w') as log:
  process=subprocess.Popen([sys.executable,__file__,'run',str(folder),runtime,build,source,batch],
      stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 pidfile.write_text(str(process.pid));print('Started test PID',process.pid,flush=True)
elif mode=='run':
 if batch=='chunk-suite':
  command=[sys.executable,str(Path(source)/'test2026/metal/benchmark_chunk.py'),runtime,build,source,str(folder)]
 elif batch=='optimization-suite':
  command=[sys.executable,str(Path(source)/'test2026/metal/benchmark_optimization.py'),runtime,build,source,str(folder)]
 elif batch.startswith('dense-download:'):
  name=batch.split(':')[1]
  command=[sys.executable,str(Path(source)/'test2026/metal/download_dense_checkpoint.py'),str(Path(source)/'test2026/metal/dense_models'/(name+'.json')),str(Path(source).parent/'models'/name)]
 elif batch=='download':
  command=[sys.executable,str(Path(source)/'test2026/pi5b/download_full_checkpoint.py'),
      str(Path(source).parent/'models/Qwen3.5-0.8B')]
 elif batch=='modelscope-download':
  command=[sys.executable,str(Path(source)/'test2026/metal/download_modelscope_checkpoint.py'),
      str(Path(source).parent/'models/Qwen3.5-0.8B')]
 elif batch.startswith('attention'):
  command=[sys.executable,str(Path(source)/'test2026/metal/test_attention_graph.py'),runtime,build,source]
  if ':' in batch:command.append(batch.split(':')[1])
 elif batch=='chunk':
  command=[sys.executable,str(Path(source)/'test2026/metal/test_chunk_graph.py'),runtime,build,source,str(Path(source).parent/'models/Qwen3.5-0.8B')]
 elif batch.startswith('dense-reference:'):
  command=[sys.executable,str(Path(source)/'test2026/metal/test_dense_reference.py'),runtime,build,source,str(Path(source).parent/'models'/batch.split(':')[1])]
 elif batch.startswith('dense-text:'):
  parts=batch.split(':');name=parts[1];prompt='What is the capital of France? Answer briefly.' if name=='Qwen3-0.6B' else 'Write a Python function add(a, b) that returns their sum. Output only code.'
  command=[sys.executable,str(Path(source)/'test2026/metal/benchmark_dense_text.py'),runtime,build,source,str(Path(source).parent/'models'/name),parts[2],'64',prompt,parts[3]]
 elif batch=='pretrained':
  command=[sys.executable,str(Path(source)/'test2026/metal/test_pretrained_graph.py'),runtime,build,source,
      str(Path(source).parent/'models/Qwen3.5-0.8B')]
 elif batch=='model':
  command=[sys.executable,str(Path(source)/'test2026/metal/test_decoder_model.py'),runtime,build,source]
 elif batch=='pipeline':
  command=[sys.executable,str(Path(source)/'test2026/metal/test_image_pipeline.py'),runtime,build,source]
 elif batch=='vision':
  command=[sys.executable,str(Path(source)/'test2026/metal/test_vision_graph.py'),runtime,build,source]
 elif batch=='pretrained-vision':
  command=[sys.executable,str(Path(source)/'test2026/metal/test_pretrained_vision_graph.py'),runtime,build,source,
      str(Path(source).parent/'models/Qwen3.5-0.8B')]
 elif batch.startswith('image-chunk:'):
  parts=batch.split(':');command=[sys.executable,str(Path(source)/'test2026/metal/benchmark_image.py'),runtime,build,source,str(Path(source).parent/'models/Qwen3.5-0.8B'),str(Path(source).parent/'images'/('RealWorld-04.batch'+parts[1]+'.json')),'32',parts[2]]
 elif batch.startswith('image:') or batch.startswith('image-long:'):
  command=[sys.executable,str(Path(source)/'test2026/metal/benchmark_image.py'),runtime,build,source,
      str(Path(source).parent/'models/Qwen3.5-0.8B'),str(Path(source).parent/'images'/('RealWorld-04.batch'+batch.split(':')[1]+'.json'))]
  if batch.startswith('image-long:'):command.append('32')
 elif batch.startswith('text-chunk:'):
  parts=batch.split(':');command=[sys.executable,str(Path(source)/'test2026/metal/benchmark_text.py'),runtime,build,source,str(Path(source).parent/'models/Qwen3.5-0.8B'),parts[1],'32','Explain three benefits of running language models locally.',parts[2]]
 elif batch.startswith('text:') or batch.startswith('text-long:'):
  command=[sys.executable,str(Path(source)/'test2026/metal/benchmark_text.py'),runtime,build,source,
      str(Path(source).parent/'models/Qwen3.5-0.8B'),batch.split(':')[1]]
  if batch.startswith('text-long:'):command.extend(['32','Explain three benefits of running language models locally.'])
 else:
  command=[sys.executable,str(Path(source)/'test2026/metal/test_linear_graph.py'),runtime,build,source,batch]
 try:
  result=subprocess.run(command)
  exitfile.write_text(json.dumps(dict(command=command,exit_code=result.returncode)))
 except Exception as error:
  exitfile.write_text(json.dumps(dict(command=command,error=type(error).__name__,exit_code=1)))
  print('Background job failed:',type(error).__name__,flush=True);raise SystemExit(1)
else:raise ValueError('mode must be launch or run')
