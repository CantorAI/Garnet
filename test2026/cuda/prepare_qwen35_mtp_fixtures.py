"""Independent CPU PyTorch MTP fixtures; flat files in an existing directory."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root).resolve()
assert root.is_dir();manifest=root/'mtp-component-fixtures.json';assert not manifest.exists()
rng=np.random.default_rng(258315);cases=[]
def write(name,value):
    value=np.asarray(value,dtype=np.float32);path=root/(name+'.f32');assert not path.exists()
    value.tofile(path);return {'path':str(path),'shape':list(value.shape)}
for geometry,heads,kv,profiles in [('2b',8,2,[(1,4),(2,4),(4,4),(8,4),(1,16)]),('4b',16,4,[(1,4),(8,4)])]:
    h,ff,dim=8,16,256
    c={'hidden_size':h,'num_attention_heads':heads,'num_key_value_heads':kv,'head_dim':dim,
        'rms_norm_eps':1e-6,'mtp_num_hidden_layers':1,'rope_parameters':{'partial_rotary_factor':.25,
        'rope_theta':10000000.,'mrope_section':[11,11,10]}}
    shapes={'fc.weight':(h,2*h),'pre_fc_norm_embedding.weight':(h,),'pre_fc_norm_hidden.weight':(h,),
        'norm.weight':(h,),'layers.0.input_layernorm.weight':(h,),'layers.0.post_attention_layernorm.weight':(h,),
        'layers.0.self_attn.q_proj.weight':(heads*2*dim,h),'layers.0.self_attn.k_proj.weight':(kv*dim,h),
        'layers.0.self_attn.v_proj.weight':(kv*dim,h),'layers.0.self_attn.o_proj.weight':(h,heads*dim),
        'layers.0.self_attn.q_norm.weight':(dim,),'layers.0.self_attn.k_norm.weight':(dim,),
        'layers.0.mlp.gate_proj.weight':(ff,h),'layers.0.mlp.up_proj.weight':(ff,h),'layers.0.mlp.down_proj.weight':(h,ff)}
    w={k:torch.tensor(rng.normal(0,.08,s),dtype=torch.bfloat16) for k,s in shapes.items()}
    checkpoint=root/f'mtp-component-{geometry}-weights.safetensors';assert not checkpoint.exists()
    table=torch.tensor(rng.normal(0,.15,(64,h)),dtype=torch.bfloat16)
    save_file({**{'mtp.'+k:v for k,v in w.items()},'model.language_model.embed_tokens.weight':table},str(checkpoint))
    norm=lambda x,k:x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*(1+w[k].float())
    project=lambda x,k:F.linear(x.to(torch.bfloat16),w[k]).float()
    for batch,chunk in profiles:
        cache=32;keys=torch.zeros(batch,kv,cache,dim);values=keys.clone();counts=[0]*batch;steps=[]
        for step in range(2):
            target=torch.tensor(rng.normal(0,.2,(batch,chunk,h)),dtype=torch.float32)
            embedding=torch.tensor(rng.normal(0,.2,(batch,chunk,h)),dtype=torch.float32)
            active=torch.tensor([[int((b+t+step)%3!=1 and not (batch==8 and b==7)) for t in range(chunk)] for b in range(batch)],dtype=torch.float32)
            if batch==1 and step==0:active.zero_()
            slots=torch.zeros(batch,chunk);positions=torch.zeros(batch,chunk,3)
            for b in range(batch):
                for t in range(chunk):
                    slot=counts[b];slots[b,t]=slot;positions[b,t]=torch.tensor([slot*2+3,slot+1,slot%3+2]);counts[b]+=int(active[b,t])
            x=project(torch.cat([norm(embedding,'pre_fc_norm_embedding.weight'),norm(target,'pre_fc_norm_hidden.weight')],-1),'fc.weight')
            x=x*active[...,None];n=norm(x,'layers.0.input_layernorm.weight')
            qg=project(n,'layers.0.self_attn.q_proj.weight').reshape(batch,chunk,heads,2*dim)
            q,gate=qg.chunk(2,-1);q=norm(q,'layers.0.self_attn.q_norm.weight')
            k=norm(project(n,'layers.0.self_attn.k_proj.weight').reshape(batch,chunk,kv,dim),'layers.0.self_attn.k_norm.weight')
            v=project(n,'layers.0.self_attn.v_proj.weight').reshape(batch,chunk,kv,dim)
            axes=[1 if i%3==1 and i<33 else 2 if i%3==2 and i<30 else 0 for i in range(32)]
            freq=torch.tensor([10000000.**(-2*i/64) for i in range(32)])
            angle=positions[...,axes]*freq;angle=torch.cat([angle,angle],-1).unsqueeze(2)
            def rotate(a):
                part=a[...,:64];left,right=part.chunk(2,-1)
                return torch.cat([part*angle.cos()+torch.cat([-right,left],-1)*angle.sin(),a[...,64:]],-1)
            q,k=rotate(q),rotate(k);cores=[]
            for t in range(chunk):
                for b in range(batch):
                    if active[b,t]:keys[b,:,int(slots[b,t])]=k[b,t];values[b,:,int(slots[b,t])]=v[b,t]
                rk=keys.repeat_interleave(heads//kv,1);rv=values.repeat_interleave(heads//kv,1)
                score=torch.einsum('bhd,bhsd->bhs',q[:,t],rk)/math.sqrt(dim)
                score=score.masked_fill(torch.arange(cache)[None,None,:]>slots[:,t,None,None],-1e30)
                core=torch.einsum('bhs,bhsd->bhd',score.softmax(-1),rv)*gate[:,t].sigmoid()
                cores.append(core.reshape(batch,1,heads*dim)*active[:,t,None,None])
            x=x+project(torch.cat(cores,1),'layers.0.self_attn.o_proj.weight')*active[...,None]
            n=norm(x,'layers.0.post_attention_layernorm.weight')
            x=x+project(F.silu(project(n,'layers.0.mlp.gate_proj.weight'))*project(n,'layers.0.mlp.up_proj.weight'),'layers.0.mlp.down_proj.weight')
            expected=norm(x,'norm.weight')
            tag=f'mtp-fixture-{geometry}-b{batch}-t{chunk}-s{step}'
            steps.append({'inputs':[write(tag+'-'+name,t.numpy()) for name,t in [('target',target),('embedding',embedding),('positions',positions),('slots',slots),('active',active)]],
                'expected':[write(tag+'-'+name,t.numpy()) for name,t in [('hidden',expected),('keys',keys),('values',values)]]})
        cases.append({'checkpoint':str(checkpoint),'config':c,'batch':batch,'chunk':chunk,'capacity':cache,'steps':steps})
manifest.write_text(json.dumps(cases,indent=2));print('QWEN35_MTP_FIXTURES_PREPARED',len(cases))
