"""Independent FP32-valued dense Qwen reference with wider dot/reduction accumulation."""
import mmap,struct,json
import numpy as np

def weights(path,layer,config):
    names=['input_layernorm.weight','post_attention_layernorm.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']
    names+=['self_attn.q_norm.weight','self_attn.k_norm.weight'] if config['model_type']=='qwen3' else ['self_attn.q_proj.bias','self_attn.k_proj.bias','self_attn.v_proj.bias']
    result={}
    with open(path,'rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
        length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);start=8+length
        for name in names:
            info=header['model.layers.'+str(layer)+'.'+name];a,b=info['data_offsets'];count=int(np.prod(info['shape']))
            if info['dtype']=='BF16':value=(np.frombuffer(mapping,dtype='<u2',count=count,offset=start+a).astype(np.uint32)<<16).view(np.float32)
            elif info['dtype']=='F32':value=np.frombuffer(mapping,dtype='<f4',count=count,offset=start+a).copy()
            else:raise ValueError('Unsupported reference weight dtype')
            result[name]=value.reshape(info['shape'])
    return result

def rms(x,w,eps):
    value=x.astype(np.float64)
    return (value/np.sqrt(np.mean(value*value,axis=-1,keepdims=True)+eps)*w.astype(np.float64)).astype(np.float32)

def project(x,w):return (x.astype(np.float64)@w.astype(np.float64).T).astype(np.float32)

def block(hidden,key,value,positions,slots,active,w,c):
    batch=len(hidden);heads=c['num_attention_heads'];kv=c['num_key_value_heads'];dim=c.get('head_dim',c['hidden_size']//heads);eps=c['rms_norm_eps']
    residual=hidden*active[:,None];x=rms(residual,w['input_layernorm.weight'],eps)
    projections=[]
    for label,count in [('q',heads),('k',kv),('v',kv)]:
        name='self_attn.'+label+'_proj';p=project(x,w[name+'.weight'])
        if c['model_type']=='qwen2':p=p+w[name+'.bias']
        p=p.reshape(batch,count,dim)
        if label!='v':
            if c['model_type']=='qwen3':p=rms(p,w['self_attn.'+label+'_norm.weight'],eps)
            frequency=(c['rope_theta']**(-np.arange(0,dim,2,dtype=np.float64)/dim)).astype(np.float32)
            angle=positions[:,0,None]*frequency[None,:];cos=np.cos(angle)[:,None,:];sin=np.sin(angle)[:,None,:]
            left,right=p[...,:dim//2],p[...,dim//2:]
            p=np.concatenate([left*cos-right*sin,right*cos+left*sin],axis=-1)
        projections.append(p)
    q,k,v=projections;key=key.copy();value=value.copy()
    for b in range(batch):
        if active[b]:key[b,:,int(slots[b])]=k[b];value[b,:,int(slots[b])]=v[b]
    keys=np.repeat(key,heads//kv,axis=1);values=np.repeat(value,heads//kv,axis=1)
    scores=np.einsum('bhd,bhsd->bhs',q,keys,optimize=False)/np.sqrt(np.float32(dim))
    mask=np.arange(key.shape[2])[None,None,:]>slots[:,None,None];scores=np.where(mask,-np.inf,scores)
    probability=np.exp(scores-np.max(scores,axis=-1,keepdims=True));probability/=np.sum(probability,axis=-1,keepdims=True);probability*=active[:,None,None]
    context=np.einsum('bhs,bhsd->bhd',probability,values,optimize=False).reshape(batch,heads*dim)
    residual=residual+project(context,w['self_attn.o_proj.weight'])
    x=rms(residual,w['post_attention_layernorm.weight'],eps);gate=project(x,w['mlp.gate_proj.weight']);up=project(x,w['mlp.up_proj.weight'])
    with np.errstate(over='ignore'):activated=gate/(1+np.exp(-gate))*up
    output=residual+project(activated,w['mlp.down_proj.weight'])
    return output.astype(np.float32),key.astype(np.float32),value.astype(np.float32)
