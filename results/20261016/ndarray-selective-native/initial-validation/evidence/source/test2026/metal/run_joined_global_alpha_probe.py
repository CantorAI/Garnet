"""Guarded trained paired-product probe; no full-model performance claim."""
import ast, hashlib, json, mmap, os, signal, struct, subprocess, sys, time, shutil
from pathlib import Path

root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False);output=out
s=root/'source/test2026/metal';statuses=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
nodes=[n for n in ast.parse((s/'run_constant_bf16_probe.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('digest','check','run')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'exclusive_runner','exec'),globals())
paths=[Path(__file__),s/'joined_global_alpha_gpu.mm',root/'source/src/backends/metal/exact_half_codec.h',s/'compare_vllm_metal.py',s/'run_constant_bf16_probe.py']
meta=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},assets={})
for p in paths:
 target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
(out/'metadata.json').write_bytes((json.dumps(meta,indent=2)+'\n').encode())
binary=out/'joined_global_alpha_gpu'
check();run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(s/'joined_global_alpha_gpu.mm'),'-o',str(binary)])
meta['binary_sha256']=digest(binary)
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 assets=root/'logs/mps-gemv-probe-20261013/jobs'/(model+'-assets');info=json.loads((assets/'manifest.json').read_text())
 checkpoint=root/'models'/model/'model.safetensors';assert digest(checkpoint)==info['checkpoint_sha256']
 members=[];raw=[]
 with checkpoint.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mapping:
  length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);base=8+length
  for ending in ('.mlp.gate_proj.weight','.mlp.up_proj.weight'):
   candidates=[c for c in info['cases'] if c['tensor'].endswith(ending)];assert len(candidates)==1
   c=candidates[0];path=assets/(c['name']+'.bin');assert digest(path)==c['sha256']
   tensor=header[c['tensor']];begin,end=tensor['data_offsets'];assert tensor['dtype']=='BF16' and tensor['shape']==[c['columns'],c['inner']]
   learned=mapping[base+begin:base+end];assert hashlib.sha256(learned).hexdigest()==c['sha256']
   members.append(c);raw.append(learned)
 assert members[0]['columns']==members[1]['columns'] and members[0]['inner']==members[1]['inner']
 joined=out/(model+'-joined.bin');joined.write_bytes(b''.join(raw))
 case=dict(model=model,tensor='joined_gate_up',path=str(joined),columns=2*members[0]['columns'],inner=members[0]['inner'],sha256=digest(joined))
 meta['assets'][model]=dict(checkpoint_sha256=info['checkpoint_sha256'],members=members,joined=case)
 (out/'metadata.json').write_bytes((json.dumps(meta,indent=2)+'\n').encode())
 manifest=out/(model+'-manifest.json');manifest.write_bytes((json.dumps(dict(cases=[case],rows=[8,16,22,24,26,32,44,48,52,64,80,128,160]),indent=2)+'\n').encode())
 check();run(model+'-gpu',[str(binary),str(manifest)])
print('Trained joined-alpha products executed; no full-model serving claim',flush=True)
