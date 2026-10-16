"""CPU full-layer exact common-scale gate/up eligibility; no GPU or inference timing."""
from pathlib import Path
import ast,hashlib,json,mmap,os,struct,sys,time
import numpy as np
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();code=1
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
try:
    prior=root/'logs/ndarray-weight-admission-20261016/report.json';assert digest(prior)=='f9505c75246b7ac43fbf1c28987104dd1e5d4c7101ae265e984af20ba246df97'
    legacy=Path(__file__).with_name('audit_ndarray_weight_admission.py');header=root/'source/src/backends/metal/exact_half_codec.h'
    assert digest(header)=='e8800f4983bb2786a7e376ca7784a7817176a5e6dd2d46c7e353116f99b2a4c9'
    nodes=[n for n in ast.parse(legacy.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='codec'];assert len(nodes)==1
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(legacy),'exec'),ns);codec=ns['codec']
    paths=[Path(__file__),legacy,header,prior];identities={p.as_posix():digest(p) for p in paths}
    for p in paths:
        dest=out/'source-snapshot'/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    (out/'pid.json').write_text(json.dumps(dict(pid=os.getpid())))
    (out/'metadata.json').write_text(json.dumps(dict(command=sys.argv,source_sha256=identities,python=sys.version,numpy=np.__version__,scope=__doc__,full_goal_met=False),indent=2))
    original=json.loads(prior.read_text());models=[]
    for previous in original['models']:
        model=previous['model'];checkpoint=root/'models'/model/'model.safetensors';assert digest(checkpoint)==previous['checkpoint_sha256']
        config=json.loads((root/'models'/model/'config.json').read_text());text=config.get('text_config',config);layers=text['num_hidden_layers'];cases={c['tensor']:c for c in previous['cases']}
        pairs=[]
        with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapped:
            size=struct.unpack_from('<Q',mapped)[0];table=json.loads(mapped[8:8+size]);base=8+size;(out/(model+'-header.json')).write_bytes(mapped[8:8+size])
            prefix='model.language_model.layers.' if model=='Qwen3.5-0.8B' else 'model.layers.'
            for layer in range(layers):
                names=[prefix+str(layer)+'.mlp.'+kind+'_proj.weight' for kind in ('gate','up')];bits=[];sides=[]
                for name in names:
                    old=cases[name];item=table[name];assert item['dtype']=='BF16' and item['shape']==[text['intermediate_size'],text['hidden_size']]
                    lo,hi=item['data_offsets'];raw=mapped[base+lo:base+hi];assert hashlib.sha256(raw).hexdigest()==old['raw_sha256']
                    words=np.frombuffer(raw,dtype='<u2').reshape(item['shape']);bits.append(words)
                    sides.append(dict(tensor=name,rows=item['shape'][0],inner=item['shape'][1],data_offset=base+lo,original_bytes=len(raw),raw_sha256=old['raw_sha256'],independent_common_scale=old['common_scale'],independent_common_ready=old['global_half_ready']))
                largest=max(int((b&32767).max()) for b in bits);exponent=(largest>>7)&255;scale=0 if largest==0 else 142-exponent
                checks=[codec(b,scale) for b in bits];admitted=all(c['admitted'] for c in checks) and (largest==0 or exponent not in (0,255)) and -64<=scale<=64
                old_eligible=all(s['independent_common_ready'] for s in sides) and sides[0]['independent_common_scale']==sides[1]['independent_common_scale']
                n,k=bits[0].shape;assert all(b.shape==(n,k) for b in bits)
                pair=dict(layer=layer,rows=n,inner=k,union_scale=scale,prior_equal_scale_eligible=old_eligible,union_admitted=admitted,sides=sides,checks=checks,
                    exact_roundtrip_values=sum(b.size for b,c in zip(bits,checks) if c['admitted']),
                    original_half_weight_payload_bytes=sum(s['original_bytes'] for s in sides),joined_half_weight_payload_bytes=n*k*4,
                    scope='Logical packed half payload estimate only; no native allocation, setup time, GPU or inference claim')
                assert pair['joined_half_weight_payload_bytes']==pair['original_half_weight_payload_bytes'];pairs.append(pair)
        model_record=dict(model=model,checkpoint=str(checkpoint),checkpoint_sha256=previous['checkpoint_sha256'],layers=layers,pairs=pairs,
            old_equal_scale_pairs=sum(p['prior_equal_scale_eligible'] for p in pairs),union_pairs=sum(p['union_admitted'] for p in pairs),
            newly_eligible_layers=[p['layer'] for p in pairs if p['union_admitted'] and not p['prior_equal_scale_eligible']],fallback_layers=[p['layer'] for p in pairs if not p['union_admitted']])
        models.append(model_record);(out/'report.json').write_text(json.dumps(dict(models=models,scope=__doc__,full_goal_met=False),indent=2)+'\n')
        print(json.dumps({k:v for k,v in model_record.items() if k not in ('pairs','checkpoint')}),flush=True)
    assert sum(m['layers'] for m in models)==76
    assert all(digest(Path(p))==sha for p,sha in identities.items());code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start)))
