"""Actual Garnet vision graph versus independent BF16-weight reference."""
import sys,json,tempfile,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi5b'))
from checkpoint_fixture import write_checkpoint
from reference_metal_vision import reference
rng=np.random.default_rng(3508)
config=dict(hidden_size=16,num_heads=2,depth=2,patch_size=2,temporal_patch_size=2,in_channels=3,
    spatial_merge_size=2,num_position_embeddings=9,out_hidden_size=12)
shapes={'patch_embed.proj.weight':(16,3,2,2,2),'patch_embed.proj.bias':(16,), 'pos_embed.weight':(9,16),
    'merger.norm.weight':(16,),'merger.norm.bias':(16,),'merger.linear_fc1.weight':(64,64),
    'merger.linear_fc1.bias':(64,),'merger.linear_fc2.weight':(12,64),'merger.linear_fc2.bias':(12,)}
for layer in range(2):
    p='blocks.'+str(layer)
    for name in ('norm1','norm2'):shapes[p+'.'+name+'.weight']=(16,);shapes[p+'.'+name+'.bias']=(16,)
    for name,rows,cols in [('attn.qkv',48,16),('attn.proj',16,16),('mlp.linear_fc1',32,16),('mlp.linear_fc2',16,32)]:
        shapes[p+'.'+name+'.weight']=(rows,cols);shapes[p+'.'+name+'.bias']=(rows,)
weights={name:rng.normal(0,.12,shape).astype(np.float32) for name,shape in shapes.items()}
for name in weights:
    if '.norm' in name and name.endswith('.weight'):weights[name]+=1
for batch,gh,gw in [(1,2,2),(1,4,6),(2,2,2),(2,4,6)]:
    with tempfile.TemporaryDirectory(prefix='garnet-vision-reference-') as temporary:
        temporary=Path(temporary);checkpoint=temporary/'weights.safetensors'
        decoded=write_checkpoint(checkpoint,weights,{name:name for name in weights},'model.visual')
        patches=rng.normal(size=(batch,gh*gw,24)).astype(np.float32)
        steps=[]
        for step in range(2):
            expected=reference(patches,decoded,config,gh,gw)
            patch_file=temporary/('patch-'+str(step)+'.f32');patches.astype('<f4').tofile(patch_file)
            steps.append(dict(input=patches.ravel().tolist(),patch_file=str(patch_file),expected=expected.ravel().tolist()))
            patches=patches.copy();patches[-1]*=-2
        fixture=dict(checkpoint=str(checkpoint),config=config,shape=list(patches.shape),grid_h=gh,grid_w=gw,steps=steps)
        path=temporary/'fixture.json';path.write_text(json.dumps(fixture))
        subprocess.run([sys.argv[1],str(Path(__file__).with_name('run_vision_graph.py')),str(path),sys.argv[2],sys.argv[3]],check=True)

