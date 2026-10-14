"""Actual trained vision products: full F64 checks and sequential ABBA timing."""
import ast,hashlib,json,mmap,os,signal,struct,subprocess,sys,time,shutil
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False);output=out
s=root/'source/test2026/metal';statuses=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
nodes=[n for n in ast.parse((s/'run_constant_bf16_probe.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('digest','check','run')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'exclusive_runner','exec'),globals())
paths=[Path(__file__),s/'vision_global_alpha_gpu.mm',root/'source/src/backends/metal/exact_half_codec.h',s/'compare_vllm_metal.py',s/'run_constant_bf16_probe.py']
meta=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},assets={})
for p in paths:
 target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
(out/'metadata.json').write_bytes((json.dumps(meta,indent=2)+'\n').encode())
binary=out/'vision_global_alpha_gpu';check()
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(s/'vision_global_alpha_gpu.mm'),'-o',str(binary)])
meta['binary_sha256']=digest(binary)
model='Qwen3.5-0.8B';checkpoint=root/'models'/model/'model.safetensors';cases=[]
with checkpoint.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mapping:
 length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);base=8+length
 # Cover every block and merger projection. Patch embeddings, norms and biases
 # retain their existing path; they are not 2D block/merger learned products.
 for name,tensor in sorted(header.items()):
  selected=name.startswith('model.visual.blocks.') or name.startswith('model.visual.merger.linear_fc')
  if not selected or not name.endswith('.weight') or len(tensor['shape'])!=2:continue
  assert tensor['dtype']=='BF16';columns,inner=tensor['shape'];begin,end=tensor['data_offsets'];raw=mapping[base+begin:base+end]
  assert len(raw)==columns*inner*2
  path=out/(name+'.bin');path.write_bytes(raw)
  cases.append(dict(model=model,tensor=name,path=str(path),columns=columns,inner=inner,sha256=digest(path),rows=[54,108] if name.startswith('model.visual.merger.') else [216,432]))
assert cases
meta['assets'][model]=dict(checkpoint_sha256=digest(checkpoint),cases=cases)
(out/'metadata.json').write_bytes((json.dumps(meta,indent=2)+'\n').encode())
manifest=out/'manifest.json';manifest.write_bytes((json.dumps(dict(cases=cases),indent=2)+'\n').encode())
check();run('vision-gpu',[str(binary),str(manifest)])
print('All trained vision products audited; excluded scales preserved; no serving claim',flush=True)
