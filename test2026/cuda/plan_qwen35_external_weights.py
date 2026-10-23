"""Checkpoint-header audit for a proposed shared projection pool; CPU only."""
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();output=root/'external-projection-weight-plan.json';assert not output.exists()
plans=[]
for size in ('2B','4B'):
    model=root/'models'/('Qwen3.5-'+size);config=json.loads((model/'config.json').read_text())['text_config'];metadata={};headers={}
    for path in sorted(model.glob('*.safetensors')):
        with path.open('rb') as stream:
            length=int.from_bytes(stream.read(8),'little');assert 0<length<=64<<20
            raw=stream.read(length);assert len(raw)==length
        headers[path.name]={'header_sha256':hashlib.sha256(raw).hexdigest(),'file_bytes':path.stat().st_size}
        for name,entry in json.loads(raw).items():
            if name=='__metadata__':continue
            assert name not in metadata;metadata[name]=entry
    expected={};h=config['hidden_size'];ff=config['intermediate_size']
    for i,kind in enumerate(config['layer_types']):
        prefix='model.language_model.layers.'+str(i)
        shapes={'mlp.gate_proj':[ff,h],'mlp.up_proj':[ff,h],'mlp.down_proj':[h,ff]}
        if kind=='linear_attention':
            kh=config['linear_num_key_heads'];vh=config['linear_num_value_heads'];kd=config['linear_key_head_dim'];vd=config['linear_value_head_dim']
            shapes.update({'linear_attn.in_proj_qkv':[2*kh*kd+vh*vd,h],'linear_attn.in_proj_z':[vh*vd,h],
                'linear_attn.in_proj_b':[vh,h],'linear_attn.in_proj_a':[vh,h],'linear_attn.out_proj':[h,vh*vd]})
        else:
            heads=config['num_attention_heads'];kv=config['num_key_value_heads'];d=config['head_dim']
            shapes.update({'self_attn.q_proj':[2*heads*d,h],'self_attn.k_proj':[kv*d,h],'self_attn.v_proj':[kv*d,h],'self_attn.o_proj':[h,heads*d]})
        for suffix,shape in shapes.items():
            name=prefix+'.'+suffix+'.weight';entry=metadata[name]
            assert entry['shape']==shape and entry['dtype']=='BF16',(name,entry,shape)
            count=shape[0]*shape[1]*2;assert entry['data_offsets'][1]-entry['data_offsets'][0]==count
            expected[name]={'shape':shape,'bytes':count}
    actual={name for name,e in metadata.items() if name.startswith('model.language_model.layers.') and len(e['shape'])==2}
    assert actual==set(expected),(actual-set(expected),set(expected)-actual)
    plans.append({'model':'Qwen3.5-'+size,'header_sources':headers,'matrix_count':len(expected),
        'one_pool_projection_bytes':sum(e['bytes'] for e in expected.values()),'matrices':expected,
        'scope':'Header-audited projection payload only. Proposed pool, not implemented/measured GPU memory or speed. Embedding, vision, small norms/conv and runtime buffers are separate.'})
output.write_text(json.dumps(plans,indent=2));print([(p['model'],p['matrix_count'],p['one_pool_projection_bytes']) for p in plans])
