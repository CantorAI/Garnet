"""Exclusive residual-add/RMS fusion, original float math and both outputs."""
import ast,hashlib,json,mmap,os,signal,struct,subprocess,sys,time,shutil
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
s=root/'source/test2026/metal';output=out;statuses=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
nodes=[n for n in ast.parse((s/'run_indirect_bf16_projection.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('digest','check','run')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'exclusive_runner','exec'),globals())
native=root/'source/src/backends/metal/metal_context.mm';paths=[native,s/'residual_norm_gpu.mm',Path(__file__),s/'run_indirect_bf16_projection.py',s/'compare_vllm_metal.py']
metadata=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},assets={},
 versions=dict(python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True)),
 hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')})
for p in paths:
 target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
text=native.read_text()
def function(name):
 start=text.index('kernel void '+name+'(');brace=text.index('{',start);depth=1;end=brace+1
 while depth:depth+=(text[end]=='{')-(text[end]=='}');end+=1
 return text[start:end]+'\n'
fused=function('rms_norm').replace('kernel void rms_norm(','kernel void residual_norm(')
fused=fused.replace('device const float* weight [[buffer(1)]],','device const float* b [[buffer(1)]],device const float* weight [[buffer(2)]],')
fused=fused.replace('device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],','device float* residual [[buffer(3)]],device float* c [[buffer(4)]],constant Params& p [[buffer(5)]],')
fused=fused.replace('float x=a[row*p.aStride+j];sum+=x*x;','float x=a[row*p.aStride+j]+b[row*p.cStride+j];residual[row*p.cStride+j]=x;sum+=x*x;')
fused=fused.replace('a[row*p.aStride+j]*inverse','residual[row*p.cStride+j]*inverse')
# Preserve the residual's rounded FP32 value before the normalization reads it.
fused=fused.replace('threadgroup_barrier(mem_flags::mem_threadgroup);','threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);')
shader=out/'residual-norm.metal';shader.write_bytes(('#include <metal_stdlib>\nusing namespace metal;\nstruct Params{ulong rows,columns,aStride,bStride,cStride;float epsilon,offset;};\n'+function('add_matrix')+function('rms_norm')+fused).encode());metadata['shader_sha256']=digest(shader)
cases=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 checkpoint=root/'models'/model/'model.safetensors';config=json.loads((checkpoint.parent/'config.json').read_text());config=config.get('text_config',config)
 with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
  length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);base=8+length
  names=[name for name in header if name.startswith('model.') and name.endswith(('layers.0.input_layernorm.weight','layers.0.post_attention_layernorm.weight'))]
  assert len(names)==2
  for index,name in enumerate(sorted(names)):
   tensor=header[name];assert tensor['dtype']=='BF16' and len(tensor['shape'])==1
   begin,end=tensor['data_offsets'];raw=mapping[base+begin:base+end];values=[]
   for (bits,) in struct.iter_unpack('<H',raw):values.append(struct.unpack('<f',struct.pack('<I',bits<<16))[0])
   target=out/(model+'-norm'+str(index)+'.bin');target.write_bytes(struct.pack('<'+'f'*len(values),*values))
   cases.append(dict(model=model,tensor=name,columns=len(values),offset=int(model=='Qwen3.5-0.8B'),epsilon=config['rms_norm_eps'],path=str(target)))
   metadata['assets'][model+':'+name]=dict(checkpoint_sha256=digest(checkpoint),canonical_bf16_sha256=hashlib.sha256(raw).hexdigest(),expanded_fp32_sha256=digest(target),columns=len(values))
 check()
assetfile=out/'norm-assets.json';assetfile.write_text(json.dumps(dict(cases=cases),indent=2)+'\n');metadata['asset_manifest_sha256']=digest(assetfile)
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
binary=out/'residual_norm_gpu';run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(s/'residual_norm_gpu.mm'),'-o',str(binary)])
metadata['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
run('gpu',[str(binary),str(shader),str(assetfile)])
print('Residual add and RMS fused component passed exact fresh-input, both-output, padding and FP64 gates',flush=True)
