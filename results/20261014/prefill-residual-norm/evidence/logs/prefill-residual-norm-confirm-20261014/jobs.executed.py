'Exclusive packed prefill Add/Layout/RMS probe; preserve all three outputs.'
import ast,hashlib,json,mmap,os,signal,struct,subprocess,sys,time,shutil
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
s=root/'source/test2026/metal';output=out;statuses=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
nodes=[n for n in ast.parse((s/'run_indirect_bf16_projection.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('digest','check','run')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'exclusive_runner','exec'),globals())
native=root/'source/src/backends/metal/metal_context.mm';paths=[native,s/'residual_norm_gpu.mm',s/'make_prefill_residual_norm_probe.py',s/'run_residual_norm_probe.py',Path(__file__),s/'run_indirect_bf16_projection.py',s/'compare_vllm_metal.py']
metadata=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},assets={},
 versions=dict(python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True)),
 hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')})
for p in paths:
 target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
shader=out/'prefill-residual-norm.metal';probe=out/'prefill_residual_norm_gpu.mm'
run('generate',[sys.executable,str(s/'make_prefill_residual_norm_probe.py'),str(native),str(shader),str(probe)])
metadata['shader_sha256']=digest(shader)
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
binary=out/'prefill_residual_norm_gpu';run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(probe),'-o',str(binary)])
metadata['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
run('gpu',[str(binary),str(shader),str(assetfile)])
print('Packed residual, layout and RMS component passed exact fresh-input, three-output, padding and FP64 gates',flush=True)
