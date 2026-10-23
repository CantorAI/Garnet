"""Prepare standalone native MTP inputs from a saved target trace, CPU only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('trace');p.add_argument('prefix');a=p.parse_args()
root=Path(a.checkpoint).resolve();source=Path(a.trace).resolve();prefix=Path(a.prefix).resolve()
assert prefix.parent.is_dir() and not Path(str(prefix)+'.json').exists()
data=json.loads(source.read_text());config=json.loads((root/'config.json').read_text())['text_config']
weight=None
for file in root.glob('*.safetensors'):
    with safe_open(str(file),framework='pt',device='cpu') as f:
        if 'model.language_model.norm.weight' in f.keys():weight=f.get_tensor('model.language_model.norm.weight').float();break
assert weight is not None
hidden=torch.tensor([r['target_hidden_pre_final_norm'] for r in data['rows']])
hidden=hidden*torch.rsqrt(hidden.square().mean(-1,keepdim=True)+config['rms_norm_eps'])*(1+weight)
target=hidden[:-1].numpy();embedding=np.asarray([r['input_embedding'] for r in data['rows'][1:]],dtype=np.float32)
positions=np.asarray([r['position'] for r in data['rows'][1:]],dtype=np.float32)
chunk=16;steps=[]
def write(tag,array):
    path=Path(str(prefix)+'-'+tag+'.f32');assert not path.exists();array=np.asarray(array,dtype=np.float32);array.tofile(path)
    return {'path':str(path),'shape':list(array.shape),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
for start in range(0,len(target),chunk):
    count=min(chunk,len(target)-start)
    t=np.zeros((1,chunk,config['hidden_size']),np.float32);e=t.copy();p0=np.zeros((1,chunk,3),np.float32)
    slots=np.zeros((1,chunk),np.float32);active=slots.copy()
    t[0,:count]=target[start:start+count];e[0,:count]=embedding[start:start+count];p0[0,:count]=positions[start:start+count]
    slots[0]=np.arange(start,start+chunk);active[0,:count]=1
    steps.append({'start':start,'count':count,'inputs':[write(f's{start}-{name}',v) for name,v in [('target',t),('embedding',e),('positions',p0),('slots',slots),('active',active)]]})
manifest={'checkpoint':str(root),'config':config,'batch':1,'capacity':512,'chunk':chunk,
    'source_trace':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'steps':steps,
    'prompt_tokens':len(data['prompt_token_ids'][0]),'generated_token_ids':data['generated_token_ids'][0],
    'target_normalization':'Independent CPU PyTorch FP32 final RMSNorm over traced pre-final-norm rows',
    'scope':'Teacher-forced MTP component input only; not speculative serving or timing'}
Path(str(prefix)+'.json').write_text(json.dumps(manifest,indent=2));print('QWEN35_MTP_LEARNED_INPUTS_PREPARED',len(steps),len(target))
