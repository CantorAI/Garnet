import json,mmap,struct,subprocess,sys
from pathlib import Path
import numpy as np
root=Path('/Users/zsdx/garnet-metal-investigation');output=root/'logs/head-bound-feasibility-20261012';output.mkdir(parents=True,exist_ok=True)
models=[('Qwen3-0.6B','What is the capital of France? Answer briefly.'),('Qwen2.5-Coder-0.5B-Instruct','Write a Python function add(a, b) that returns their sum. Output only code.'),('Qwen3.5-0.8B','What is the capital of France? Answer briefly.')]
summary=[]
for name,prompt in models:
 folder=output/name;folder.mkdir(exist_ok=True);model=root/'models'/name
 command=['/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3',str(root/'source/test2026/metal/probe_head_bound_inputs.py'),str(root/'build-state-handoff'),str(root/'source'),str(model),prompt]
 print('Feasibility-only native command',json.dumps(command),flush=True)
 with (folder/'output.log').open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 (folder/'exit.json').write_text(json.dumps(dict(command=command,exit_code=result.returncode),indent=2)+'\n')
 assert result.returncode==0
 records=[json.loads(line.split(' ',1)[1]) for line in (folder/'output.log').read_text().splitlines() if line.startswith('head_bound_input ')]
 assert records
 config=json.loads((model/'config.json').read_text());vl='text_config' in config
 if vl:config=config['text_config']
 weight='model.language_model.embed_tokens.weight' if vl else 'model.embed_tokens.weight' if config.get('tie_word_embeddings') else 'lm_head.weight'
 norm='model.language_model.norm.weight' if vl else 'model.norm.weight'
 with (model/'model.safetensors').open('rb') as stream:
  length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length));base=8+length
  with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
   def tensor(name):
    info=header[name];count=int(np.prod(info['shape']));offset=base+info['data_offsets'][0]
    if info['dtype']=='BF16':return (np.frombuffer(mapping,dtype='<u2',count=count,offset=offset).astype(np.uint32)<<16).view(np.float32).reshape(info['shape'])
    assert info['dtype']=='F32';return np.array(np.frombuffer(mapping,dtype='<f4',count=count,offset=offset).reshape(info['shape']),copy=True)
   nw=tensor(norm).astype(np.float64)
   info=header[weight];assert info['dtype']=='BF16';rows,columns=info['shape'];norms=np.empty(rows,np.float64)
   for start in range(0,rows,512):
    count=min(512,rows-start)*columns;bits=np.frombuffer(mapping,dtype='<u2',count=count,offset=base+info['data_offsets'][0]+start*columns*2).astype(np.uint32)
    values=(bits<<16).view(np.float32).reshape(-1,columns).astype(np.float64);norms[start:start+len(values)]=np.linalg.norm(values,axis=1)
   del bits,values
 samples=[]
 for record in records:
  hidden=np.asarray(record['hidden'],np.float64);hidden=hidden/np.sqrt(np.mean(hidden**2)+config['rms_norm_eps'])*(nw+(1 if vl else 0))
  magnitude=float(np.linalg.norm(hidden));bound=norms*magnitude*1.001
  energy=np.sort(hidden**2)[::-1];total=float(energy.sum())
  samples.append(dict(step=record['step'],prediction=record['prediction'],normalized_hidden_norm=magnitude,
   full_max=record['full_max'],seed_max=record['seed_max'],ideal_bound_prunable_fraction=float(np.mean(bound<record['full_max'])),
   fixed_seed_bound_prunable_fraction=float(np.mean(bound<record['seed_max'])),top8_energy_fraction=float(energy[:8].sum()/total),top32_energy_fraction=float(energy[:32].sum()/total)))
 report=dict(model=name,scope='Sampled feasibility only, approximate CPU head normalization with 0.1% margin; no pruning kernel or speed gain demonstrated',row_norm_quantiles=np.quantile(norms,[0,.1,.5,.9,1]).tolist(),samples=samples)
 (folder/'feasibility.json').write_text(json.dumps(report,indent=2)+'\n');summary.append(report)
 print(name,'fixed-seed prunable fraction range',min(s['fixed_seed_bound_prunable_fraction'] for s in samples),max(s['fixed_seed_bound_prunable_fraction'] for s in samples),flush=True)
(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
