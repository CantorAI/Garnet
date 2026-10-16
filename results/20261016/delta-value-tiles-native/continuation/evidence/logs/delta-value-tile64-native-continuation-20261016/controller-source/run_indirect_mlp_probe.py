"""Exclusive gate for recorded four-kernel MLPs with original learned values."""
import ast,hashlib,json,mmap,os,signal,struct,subprocess,sys,time,shutil
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
s=root/'source/test2026/metal';output=out;statuses=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
nodes=[n for n in ast.parse((s/'run_indirect_bf16_projection.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('digest','check','run')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'exclusive_runner','exec'),globals())
native=root/'source/src/backends/metal/metal_context.mm';paths=[native,s/'indirect_mlp_probe.mm',Path(__file__),s/'run_indirect_bf16_projection.py',s/'compare_vllm_metal.py']
metadata=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},assets={},
    versions=dict(python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)),
    hardware={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')})
for p in paths:
 target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
text=native.read_text();shader=out/'native-mlp.metal'
projection=text[text.index('kernel void bf16_decode('):text.index('kernel void bf16_decode_small(')]
pointwise=text[text.index('kernel void silu_mul('):text.index('kernel void add_matrix(')]
shader.write_bytes(('#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params{ulong rows,inner,columns,aStride,oStride;};\nstruct Params{ulong rows,columns,aStride,bStride,cStride;float epsilon,offset;};\n'+projection+pointwise).encode())
metadata['shader_sha256']=digest(shader);models=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 assets=root/'logs/mps-gemv-probe-20261013/jobs'/(model+'-assets');info=json.loads((assets/'manifest.json').read_text());checkpoint=root/'models'/model/'model.safetensors'
 assert digest(checkpoint)==info['checkpoint_sha256']
 selected={key:next(c for c in info['cases'] if c['tensor'].endswith('.mlp.'+key+'_proj.weight')) for key in ('gate','up','down')}
 hidden=selected['gate']['inner'];intermediate=selected['gate']['columns']
 assert (selected['up']['inner'],selected['up']['columns'])==(hidden,intermediate)
 assert (selected['down']['inner'],selected['down']['columns'])==(intermediate,hidden)
 with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
  length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);base=8+length
  for case in selected.values():
   assert digest(assets/(case['name']+'.bin'))==case['sha256'];tensor=header[case['tensor']]
   assert tensor['dtype']=='BF16' and tensor['shape']==[case['columns'],case['inner']]
   start,end=tensor['data_offsets'];assert hashlib.sha256(mapping[base+start:base+end]).hexdigest()==case['sha256']
 metadata['assets'][model]=dict(checkpoint_sha256=info['checkpoint_sha256'],cases=selected)
 models.append(dict(model=model,hidden=hidden,intermediate=intermediate,**{key:str(assets/(c['name']+'.bin')) for key,c in selected.items()}))
 check()
assetfile=out/'mlp-assets.json';assetfile.write_text(json.dumps(dict(models=models),indent=2)+'\n')
metadata['mlp_assets_sha256']=digest(assetfile);(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
binary=out/'indirect_mlp_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(s/'indirect_mlp_probe.mm'),'-o',str(binary)])
metadata['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
run('gpu',[str(binary),str(shader),str(assetfile)])
print('Four-kernel MLP serial and independent gate/up plans passed fresh-input intermediate/padding and independent numerical checks',flush=True)
