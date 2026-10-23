"""Independent two-layer CUDA decoder trajectory with ragged active rows."""
import json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from qwen35_step_reference import linear,attention
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
base=Path(sys.argv[2]).resolve();rng=np.random.default_rng(936)
batch=int(sys.argv[3]) if len(sys.argv)>3 else 2
delta_dim=int(sys.argv[4]) if len(sys.argv)>4 else 4
value_heads=int(sys.argv[5]) if len(sys.argv)>5 else 4
assert batch in (1,2,4,8) and delta_dim in (4,128) and value_heads in (2,4)
weights=load_file(str(base/'linear-fixtures/bfloat16/model.safetensors'))
attention_real=len(sys.argv)>6 and sys.argv[6]=='real-attention'
attention_directory='attention-real-fixtures' if attention_real else 'attention-fixtures'
heads,kv,dim,rotary,sections,theta=(16,4,256,64,[11,11,10],10000000.) if attention_real else (4,2,8,8,[2,1,1],10000.)
# The attention fixture uses layer0; this decoder places it after linear layer0.
weights.update({k.replace('layers.0.','layers.1.'):v for k,v in load_file(str(base/attention_directory/'bfloat16/model.safetensors')).items()})
if delta_dim!=4 or value_heads!=4:
    channels=4*delta_dim+value_heads*delta_dim
    for suffix,shape in [('in_proj_qkv.weight',(channels,8)),('in_proj_z.weight',(value_heads*delta_dim,8)),
        ('in_proj_b.weight',(value_heads,8)),('in_proj_a.weight',(value_heads,8)),('conv1d.weight',(channels,1,4)),
        ('out_proj.weight',(8,value_heads*delta_dim)),('dt_bias',(value_heads,)),('A_log',(value_heads,)),('norm.weight',(delta_dim,))]:
        data=rng.normal(0,.1,shape).astype(np.float32)
        if suffix=='norm.weight':data+=1
        weights['model.language_model.layers.0.linear_attn.'+suffix]=torch.from_numpy(data).to(torch.bfloat16)
weights['model.language_model.embed_tokens.weight']=torch.from_numpy(rng.normal(0,.15,(32,8)).astype(np.float32)).to(torch.bfloat16)
for layer in range(2):
    p='model.language_model.layers.'+str(layer)
    for n in ['input_layernorm','post_attention_layernorm']:
        weights[p+'.'+n+'.weight']=torch.from_numpy(rng.normal(0,.1,8).astype(np.float32)).to(torch.bfloat16)
    for n,s in [('gate_proj',(16,8)),('up_proj',(16,8)),('down_proj',(8,16))]:
        weights[p+'.mlp.'+n+'.weight']=torch.from_numpy(rng.normal(0,.1,s).astype(np.float32)).to(torch.bfloat16)
weights['model.language_model.norm.weight']=torch.from_numpy(rng.normal(0,.1,8).astype(np.float32)).to(torch.bfloat16)
save_file(weights,str(out/'model.safetensors'))
c={'hidden_size':8,'num_hidden_layers':2,'layer_types':['linear_attention','full_attention'],
 'linear_num_key_heads':2,'linear_num_value_heads':value_heads,'linear_key_head_dim':delta_dim,'linear_value_head_dim':delta_dim,
 'linear_conv_kernel_dim':4,'num_attention_heads':heads,'num_key_value_heads':kv,'head_dim':dim,'rms_norm_eps':1e-6,
 'rope_parameters':{'partial_rotary_factor':rotary/dim,'rope_theta':theta,'mrope_section':sections}}
capacity=32;conv=np.zeros((batch,4*delta_dim+value_heads*delta_dim,4),np.float32);state=np.zeros((batch,value_heads,delta_dim,delta_dim),np.float32)
key=np.zeros((batch,kv,capacity,dim),np.float32);value=key.copy();last=torch.zeros(batch,8);slots=np.zeros(batch,np.float32)
def norm(x,name):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*(1+weights[name].float())
lw={k:weights['model.language_model.layers.0.linear_attn.'+s] for k,s in
 {'qkv':'in_proj_qkv.weight','z':'in_proj_z.weight','b_proj':'in_proj_b.weight','a_proj':'in_proj_a.weight',
  'conv':'conv1d.weight','a_log':'A_log','dt_bias':'dt_bias','norm':'norm.weight','out_proj':'out_proj.weight'}.items()}
aw={k:weights['model.language_model.layers.1.self_attn.'+s] for k,s in
 {'q':'q_proj.weight','k':'k_proj.weight','v':'v_proj.weight','out':'o_proj.weight','q_norm':'q_norm.weight','k_norm':'k_norm.weight'}.items()}
steps=[]
for step,(tokens,active) in enumerate([([2,3],[1,0]),([5,7],[0,1]),([11,13],[1,1]),([17,19],[1,1])]):
    tokens=[(tokens[b%2]+2*(b//2))%32 for b in range(batch)]
    active=[active[b%2] for b in range(batch)]
    flag=np.array(active,np.float32);positions=np.stack([slots]*3,-1)
    x=weights['model.language_model.embed_tokens.weight'][tokens].float();layer_hidden=[]
    for idx in range(2):
        p='model.language_model.layers.'+str(idx);x=x*torch.tensor(active)[:,None]
        normalized=norm(x,p+'.input_layernorm.weight')
        if idx==0:att,conv,state=linear(normalized.numpy(),conv,state,flag,lw,2,value_heads,delta_dim,delta_dim)
        else:att,key,value=attention(normalized.numpy(),key,value,positions,slots,flag,aw,heads,kv,dim,rotary,sections,theta)
        x=x+torch.from_numpy(att);n=norm(x,p+'.post_attention_layernorm.weight')
        def proj(a,name):return F.linear(a.to(torch.bfloat16),weights[p+'.mlp.'+name+'.weight']).float()
        x=x+proj(F.silu(proj(n,'gate_proj'))*proj(n,'up_proj'),'down_proj')
        layer_hidden.append(x.ravel().tolist())
    last[torch.tensor(active,dtype=torch.bool)]=x[torch.tensor(active,dtype=torch.bool)]
    normalized=norm(last,'model.language_model.norm.weight')
    logits=F.linear(normalized.to(torch.bfloat16),weights['model.language_model.embed_tokens.weight']).float()
    slots+=flag
    steps.append({'tokens':tokens,'active':active,'predictions':logits.argmax(-1).tolist(),'hidden':last.ravel().tolist(),'slots':slots.tolist(),'layer_hidden':layer_hidden})
fixture={'checkpoint':str(out),'config':c,'batch':batch,'capacity':capacity,'steps':steps,
    'final_states':[{'state1':conv.ravel().tolist(),'state2':state.ravel().tolist()},
                    {'state1':key.ravel().tolist(),'state2':value.ravel().tolist()}]}
(out/'fixture.json').write_text(json.dumps(fixture));print(out/'fixture.json')
