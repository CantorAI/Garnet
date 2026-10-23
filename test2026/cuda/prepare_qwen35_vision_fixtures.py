"""Independent FP32 NumPy vision reference: interpolation, RoPE, attention, merger."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('reference',Path(__file__).resolve().parents[1]/'pi5b/reference_vision.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
rng=np.random.default_rng(934)
c=dict(hidden_size=16,num_heads=2,depth=2,patch_size=2,temporal_patch_size=2,in_channels=3,
 spatial_merge_size=2,num_position_embeddings=9,out_hidden_size=12)
shapes={'patch_embed.proj.weight':(16,3,2,2,2),'patch_embed.proj.bias':(16,),'pos_embed.weight':(9,16),
 'merger.norm.weight':(16,),'merger.norm.bias':(16,),'merger.linear_fc1.weight':(64,64),
 'merger.linear_fc1.bias':(64,),'merger.linear_fc2.weight':(12,64),'merger.linear_fc2.bias':(12,)}
for layer in range(2):
    p='blocks.'+str(layer)
    for name in ['norm1','norm2']:shapes[p+'.'+name+'.weight']=(16,);shapes[p+'.'+name+'.bias']=(16,)
    for name,rows,cols in [('attn.qkv',48,16),('attn.proj',16,16),('mlp.linear_fc1',32,16),('mlp.linear_fc2',16,32)]:
        shapes[p+'.'+name+'.weight']=(rows,cols);shapes[p+'.'+name+'.bias']=(rows,)
w={name:rng.normal(0,.12,s).astype(np.float32) for name,s in shapes.items()}
for name in w:
    if '.norm' in name and name.endswith('.weight'):w[name]+=1
checkpoint=root/'weights';checkpoint.mkdir(exist_ok=True)
save_file({'model.visual.'+k:torch.from_numpy(v) for k,v in w.items()},str(checkpoint/'model.safetensors'))
cases=[]
for batch,gh,gw in [(1,2,2),(2,4,6),(4,2,4),(8,2,2)]:
    steps=[]
    for step in range(2):
        patches=rng.normal(0,.2,(batch,gh*gw,24)).astype(np.float32)
        expected=ref.reference(patches,w,c,gh,gw).astype(np.float32)
        steps.append({'inputs':[{'shape':list(patches.shape),'values':patches.ravel().tolist()}],
            'expected':expected.ravel().tolist(),'output_shape':list(expected.shape)})
    cases.append({'kind':'vision','dtype':'float32','batch':batch,'checkpoint':str(checkpoint),
        'config':c,'grid_h':gh,'grid_w':gw,'steps':steps})
(root/'fixtures.json').write_text(json.dumps(cases));print('Prepared',len(cases),'FP32 vision cases')
