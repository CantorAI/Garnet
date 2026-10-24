"""Real-layer CPU oracle for Garnet's existing fused4/16-token programs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import prepare_real_layer_probe as arithmetic
from prepare_real_layer_probe import Checkpoint,dense,rms,linear,attention,mlp

def prepare(root):
    cp=Checkpoint(root);config=json.loads((root/'config.json').read_text());t=config['text_config'];width=t['hidden_size'];capacity=4096;layers=[]
    for layer in [0,3]:
        kind=t['layer_types'][layer];p='model.language_model.layers.'+str(layer)
        if kind=='linear_attention':
            channels=2*t['linear_num_key_heads']*t['linear_key_head_dim']+t['linear_num_value_heads']*t['linear_value_head_dim']
            s1=np.zeros((channels,t['linear_conv_kernel_dim']),dtype=np.float32)
            s2=np.zeros((t['linear_num_value_heads'],t['linear_key_head_dim'],t['linear_value_head_dim']),dtype=np.float32)
        else:s1=np.zeros((t['num_key_value_heads'],capacity,t['head_dim']),dtype=np.float32);s2=s1.copy()
        cases=[]
        for step in range(18):
            x=np.array([((i*17+step*23)%101-50)/50 for i in range(width)],dtype=np.float32);position=[3+step,5+step,7+step]
            normalized=rms(x,dense(cp,p+'.input_layernorm.weight'),t['rms_norm_eps'],True)
            if kind=='linear_attention':a,s1,s2=linear(cp,normalized,s1,s2,p+'.linear_attn',t)
            else:a,s1,s2=attention(cp,normalized,s1,s2,p+'.self_attn',t,position,step)
            residual=x+a;out=residual+mlp(cp,rms(residual,dense(cp,p+'.post_attention_layernorm.weight'),t['rms_norm_eps'],True),p+'.mlp')
            samples=[]
            for state in (s1,s2):
                flat=state.reshape(-1);indices=set([0,len(flat)-1]+[int(i) for i in np.linspace(0,len(flat)-1,512,dtype=int)])
                if kind=='full_attention':
                    # Include two not-yet-active slots to detect padded KV writes.
                    for head in range(t['num_key_value_heads']):
                        for slot in range(step+3):
                            for col in np.linspace(0,t['head_dim']-1,64,dtype=int):indices.add((head*capacity+slot)*t['head_dim']+int(col))
                indices=sorted(indices);samples.append({'indices':indices,'values':flat[indices].tolist()})
            cases.append({'input':x.tolist(),'positions':position,'expected_hidden':out.tolist(),'state_samples':samples})
        layers.append({'layer':layer,'kind':kind,'state_shapes':[list(s1.shape),list(s2.shape)],'cases':cases})
    return {'layers':layers,'capacity':capacity,'chunk_profiles':[4,16],'steps':18,'numpy_version':np.__version__,
        'fp8_scaling':'combined F32 alpha after raw FP8 dot' if arithmetic.FP8_ALPHA_AFTER_DOT else 'separate scale multiplication before dot',
        'scope':'Actual mixed layers0/3; eighteen sequential CPU steps, active and future KV samples. Tests fused4/16 followed by padded2-active chunks, not full-model quality or performance'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--fp8-alpha-after-dot',action='store_true');a=p.parse_args()
    arithmetic.FP8_ALPHA_AFTER_DOT=a.fp8_alpha_after_dot
    a.output.write_text(json.dumps(prepare(a.checkpoint)));print(json.dumps({'oracle_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}))
