"""New prerequisite: all separate learned Q/K/V triples, including retained FP32 keys; CPU only."""
from pathlib import Path
import ast,hashlib,json,mmap,os,struct,subprocess,sys,time
import numpy as np
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();code=1
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
try:
 prior=root/'validation-audits/joined-union-half-admission-20261016/evidence/logs/joined-union-half-admission-20261016/report.json'
 legacy=Path(__file__).with_name('audit_ndarray_weight_admission.py');original=Path(__file__).with_name('joined_union_half_admission_cpu.mm');codec_header=root/'source/src/backends/metal/exact_half_codec.h'
 assert digest(codec_header)=='e8800f4983bb2786a7e376ca7784a7817176a5e6dd2d46c7e353116f99b2a4c9'
 assert json.loads((root/'validation-audits/prepared-union-mlp-learned-confirm-20261016/decision-repeat.json').read_text())['confirmed_models']==[]
 nodes=[v for v in ast.parse(legacy.read_text()).body if isinstance(v,ast.FunctionDef) and v.name=='codec'];assert len(nodes)==1
 ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(legacy),'exec'),ns);codec=ns['codec']
 paths=[Path(__file__),legacy,original,codec_header,prior];meta=dict(source_sha256={p.as_posix():digest(p) for p in paths},command=sys.argv,python=sys.version,numpy=np.__version__,scope=__doc__,full_goal_met=False)
 for p in paths:
  target=out/'source-snapshot'/p.name;target.parent.mkdir(exist_ok=True);target.write_bytes(p.read_bytes())
 (out/'pid.json').write_text(json.dumps(dict(pid=os.getpid())));(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');models=[]
 for previous in json.loads(prior.read_text())['models']:
  name=previous['model'];checkpoint=Path(previous['checkpoint']);assert digest(checkpoint)==previous['checkpoint_sha256'];groups=[];skipped=[]
  with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapped:
   length=struct.unpack_from('<Q',mapped)[0];header=json.loads(mapped[8:8+length]);base=8+length;(out/(name+'-header.json')).write_bytes(mapped[8:8+length]);prefix='model.language_model.layers.' if name=='Qwen3.5-0.8B' else 'model.layers.'
   for layer in range(previous['layers']):
    names=[prefix+str(layer)+'.self_attn.'+part+'_proj.weight' for part in ('q','k','v')]
    if names[0] not in header:
     assert name=='Qwen3.5-0.8B' and all(n not in header for n in names) and prefix+str(layer)+'.linear_attn.in_proj_qkv.weight' in header
     skipped.append(dict(layer=layer,reason='Existing combined linear-attention QKV; outside separate-triple hypothesis'));continue
    bits=[];sides=[];scales=[]
    for part,tensor in zip(('q','k','v'),names):
     item=header[tensor];assert item['dtype']=='BF16' and len(item['shape'])==2;n,k=item['shape'];lo,hi=item['data_offsets'];raw=mapped[base+lo:base+hi];assert len(raw)==n*k*2
     words=np.frombuffer(raw,dtype='<u2').reshape(n,k);bits.append(words);largest=int((words&32767).max());scales.append(0 if largest==0 else 142-((largest>>7)&255))
     sides.append(dict(tensor=tensor,part=part,rows=n,inner=k,data_offset=base+lo,original_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),retained_float_key=(part=='k' and name!='Qwen3.5-0.8B')))
    assert len({b.shape[1] for b in bits})==1
    largest=max(int((b&32767).max()) for b in bits);exponent=(largest>>7)&255;scale=0 if largest==0 else 142-exponent;checks=[codec(b,scale) for b in bits];admitted=all(v['admitted'] for v in checks) and -64<=scale<=64 and (largest==0 or exponent not in (0,255))
    groups.append(dict(layer=layer,inner=bits[0].shape[1],columns=sum(b.shape[0] for b in bits),union_scale=scale,union_admitted=admitted,independent_scales=scales,sides=sides,checks=checks,logical_extra_union_bytes=sum(b.nbytes for b in bits)))
  models.append(dict(model=name,checkpoint=str(checkpoint),checkpoint_sha256=previous['checkpoint_sha256'],layers=previous['layers'],groups=groups,skipped_layers=skipped))
  (out/'report.json').write_text(json.dumps(dict(models=models,scope=__doc__,full_goal_met=False),indent=2)+'\n');print(json.dumps(dict(model=name,groups=len(groups),admitted=sum(v['union_admitted'] for v in groups),existing_combined_layers=len(skipped))),flush=True)
 assert sum(len(v['groups']) for v in models)==58
 # Adapt only group arity/count/names; preserve original EncodeExactScaledHalf and bit roundtrip loop.
 cpp=original.read_text();assert cpp.count('side<2')==1 and cpp.count('pairs==76&&matrices==152')==1
 cpp=cpp.replace('side<2','side<3').replace('pairs==76&&matrices==152','pairs==58&&matrices==174').replace('pair','group')
 generated=out/'qkv_union_half_admission_cpu.mm';generated.write_bytes(cpp.encode());meta['generated_sha256']=digest(generated);statuses=[];binary=out/'qkv_union_half_admission_cpu'
 for name,command in [('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-I'+str(codec_header.parent),str(generated),'-framework','Foundation','-o',str(binary)]),('audit',[str(binary),str(out/'report.json')])]:
  with (out/(name+'.log')).open('w') as stream:process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
  statuses.append(dict(name=name,command=command,exit_code=process.returncode));(out/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n');assert process.returncode==0
 meta['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 records=[json.loads(line) for line in (out/'audit.log').read_text().splitlines()];assert len(records)==59 and records[-1]['groups']==58 and records[-1]['matrices']==174
 assert all(digest(Path(p))==sha for p,sha in meta['source_sha256'].items());print(json.dumps(records[-1]),flush=True);code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start)))
