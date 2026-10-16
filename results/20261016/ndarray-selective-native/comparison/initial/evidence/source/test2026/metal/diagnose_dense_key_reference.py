"""Reproduce the rejected dense profile and attribute errors stage by stage.

Arguments: investigation root, new evidence directory. Diagnostic only;
no tolerance is changed and no inference speed is claimed.
"""
import ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from reference_dense import weights,rms,project
root=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
build=root/(sys.argv[3] if len(sys.argv)>3 else 'build-linear-chunk')
scripts=root/'source/test2026/metal';background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
tree=ast.parse((scripts/'compare_vllm_metal.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
assert len(functions)==3
exec(compile(ast.Module(body=functions,type_ignores=[]),'owned_comparison_guard','exec'),globals())
guard();(output/'before-processes.txt').write_text(snapshot())
model=root/'models/Qwen3-0.6B';checkpoint=model/'model.safetensors';c=json.loads((model/'config.json').read_text())
c['head_dim']=c.get('head_dim',c['hidden_size']//c['num_attention_heads'])
batch=2;chunk=32;capacity=256;d=c['hidden_size'];kv=c['num_key_value_heads'];dim=c['head_dim'];eps=c['rms_norm_eps']
rng=np.random.default_rng(765)
# Match the controller's RNG consumption before its first batch-2 chunk-32 case.
for earlier in (32,64):
    for _ in range(3):rng.normal(size=(1,earlier,d))
hidden=rng.normal(size=(batch,chunk,d)).astype(np.float32)
active=np.ones((batch,chunk),np.float32);active[1,0]=0
slots=np.zeros((batch,chunk),np.float32);positions=np.zeros((batch,chunk,3),np.float32)
for b in range(batch):
    cursor=0
    for t in range(chunk):slots[b,t]=cursor;positions[b,t]=cursor;cursor+=active[b,t]
cache=np.zeros((batch,kv,capacity,dim),np.float32)
inputs=[hidden,cache,cache.copy(),positions,slots,active];paths=[]
for i,value in enumerate(inputs):
    path=output/('input-'+str(i)+'.f32');value.astype('<f4').tofile(path);paths.append(str(path))
flat=(hidden*active[:,:,None]).reshape(batch*chunk,d)
flatpath=output/'flat-input.f32';flat.tofile(flatpath)
fixture=dict(config=c,batch=batch,chunk=chunk,capacity=capacity,checkpoint=str(checkpoint),shapes=[list(v.shape) for v in inputs],inputs=paths,flat_input=str(flatpath),output=str(output/'actual.json'))
(output/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
command=['/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3',str(scripts/'run_dense_key_diagnostic.py'),str(output/'fixture.json'),str(build),str(root/'source')]
with (output/'worker.log').open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
(output/'status.json').write_text(json.dumps(dict(command=command,exit_code=result.returncode),indent=2)+'\n')
assert result.returncode==0
guard();(output/'after-processes.txt').write_text(snapshot())
w=weights(checkpoint,0,c);actual=json.loads((output/'actual.json').read_text())
normal=np.asarray(actual['normalized'],np.float32).reshape(batch*chunk,d)
projection=np.asarray(actual['projected'],np.float32).reshape(batch,chunk,kv,dim)
keys=np.asarray(actual['key_cache'],np.float32).reshape(batch,kv,capacity,dim)
reference_norm=rms(flat,w['input_layernorm.weight'],eps)
reference_projection=project(reference_norm,w['self_attn.k_proj.weight']).reshape(batch,chunk,kv,dim)
from_gpu_norm=project(normal,w['self_attn.k_proj.weight']).reshape(batch,chunk,kv,dim)
ideal_norm=flat.astype(np.float64)
ideal_norm=ideal_norm/np.sqrt(np.mean(ideal_norm**2,axis=-1,keepdims=True)+eps)*w['input_layernorm.weight'].astype(np.float64)
ideal_projection=(ideal_norm@w['self_attn.k_proj.weight'].astype(np.float64).T).reshape(batch,chunk,kv,dim)
frequency=(c['rope_theta']**(-np.arange(0,dim,2,dtype=np.float64)/dim)).astype(np.float32)
angle=positions[:,:,0,None]*frequency[None,None,:]
cos=np.cos(angle)[:,:,None,:];sin=np.sin(angle)[:,:,None,:]
def rotated(value,wide=False):
    if wide:
        p=value.astype(np.float64);p=p/np.sqrt(np.mean(p*p,axis=-1,keepdims=True)+eps)*w['self_attn.k_norm.weight'].astype(np.float64)
    else:p=rms(value,w['self_attn.k_norm.weight'],eps)
    left,right=p[...,:dim//2],p[...,dim//2:]
    return np.concatenate([left*cos-right*sin,right*cos+left*sin],axis=-1)
reference_key=rotated(reference_projection);from_gpu_projection=rotated(projection);ideal_key=rotated(ideal_projection,True)
obtained=np.zeros_like(reference_key)
for b in range(batch):
    for t in range(chunk):
        if active[b,t]:obtained[b,t]=keys[b,:,int(slots[b,t])]
mask=active.astype(bool)
def error(a,b):
    delta=a.astype(np.float64)-b.astype(np.float64)
    return dict(max_abs=float(np.max(np.abs(delta))),l2=float(np.linalg.norm(delta)),relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(b.astype(np.float64)),1e-30)))
stages=dict(input_norm=error(normal,reference_norm),projection_total=error(projection[mask],reference_projection[mask]),
 projection_given_gpu_norm=error(projection[mask],from_gpu_norm[mask]),key_total=error(obtained[mask],reference_key[mask]),
 key_given_gpu_projection=error(obtained[mask],from_gpu_projection[mask]),gpu_key_vs_continuous_f64=error(obtained[mask],ideal_key[mask]),
 reference_key_vs_continuous_f64=error(reference_key[mask],ideal_key[mask]))
left=obtained[...,:dim//2]-reference_key[...,:dim//2];right=obtained[...,dim//2:]-reference_key[...,dim//2:]
magnitude=np.hypot(reference_key[...,:dim//2],reference_key[...,dim//2:])
ratio=np.hypot(left,right)/(5e-6+5e-5*magnitude);ratio[~mask]=0
index=tuple(int(v) for v in np.unravel_index(np.argmax(ratio),ratio.shape));b,t,h,j=index
pair=dict(batch=b,token=t,head=h,dimension=j,slot=int(slots[b,t]),worst_ratio=float(ratio[index]))
for label,value in [('gpu',obtained),('reference',reference_key),('from_gpu_projection',from_gpu_projection),('continuous_f64',ideal_key)]:pair[label]=[float(value[b,t,h,j]),float(value[b,t,h,j+dim//2])]
summary=dict(scope='Numerical attribution only; existing acceptance tolerance unchanged',model=model.name,batch=batch,chunk=chunk,stages=stages,worst_key_pair=pair,
 native_build=str(build),native_library_sha256=hashlib.file_digest((build/'libgarnet.dylib').open('rb'),'sha256').hexdigest(),
 input_sha256={Path(p).name:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest() for p in paths},runner_sha256=hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest())
(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
