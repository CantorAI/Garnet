"""Independent checks across vision, mixed decoder layers, and GPU head.

Small BF16 fixtures validate serving boundaries; they do not replace the
required pretrained image/text executions and performance measurements.
"""
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi5b'))
from checkpoint_fixture import write_checkpoint
from reference_decoder import linear_block,attention_block,rms_norm
from reference_metal_vision import reference
rng=np.random.default_rng(357)
common=dict(input_norm='input_layernorm.weight',post_norm='post_attention_layernorm.weight',
    mlp_gate='mlp.gate_proj.weight',mlp_up='mlp.up_proj.weight',mlp_down='mlp.down_proj.weight')
linear_names=dict(qkv='linear_attn.in_proj_qkv.weight',z='linear_attn.in_proj_z.weight',
    b_proj='linear_attn.in_proj_b.weight',a_proj='linear_attn.in_proj_a.weight',conv='linear_attn.conv1d.weight',
    a_log='linear_attn.A_log',dt_bias='linear_attn.dt_bias',norm='linear_attn.norm.weight',out_proj='linear_attn.out_proj.weight')
full_names=dict(q='self_attn.q_proj.weight',k='self_attn.k_proj.weight',v='self_attn.v_proj.weight',
    out='self_attn.o_proj.weight',q_norm='self_attn.q_norm.weight',k_norm='self_attn.k_norm.weight')
for batch in (1,2):
    d,kh,vh,kd,vd,kernel,heads,kv,dim,vocab,capacity=12,1,2,7,5,4,2,1,8,19,16
    channels=2*kh*kd+vh*vd
    c=dict(hidden_size=d,layer_types=['linear_attention','full_attention'],rms_norm_eps=1e-6,
        linear_num_key_heads=kh,linear_num_value_heads=vh,linear_key_head_dim=kd,linear_value_head_dim=vd,
        linear_conv_kernel_dim=kernel,num_attention_heads=heads,num_key_value_heads=kv,head_dim=dim,
        rope_parameters=dict(partial_rotary_factor=1,rope_theta=10000000,mrope_section=[2,1,1]))
    vc=dict(hidden_size=16,num_heads=2,depth=2,patch_size=2,temporal_patch_size=2,in_channels=3,
        spatial_merge_size=2,num_position_embeddings=9,out_hidden_size=d)
    weights={};suffixes={};groups={};f32=[]
    cs=dict(input_norm=(d,),post_norm=(d,),mlp_gate=(17,d),mlp_up=(17,d),mlp_down=(d,17))
    ls=dict(qkv=(channels,d),z=(vh*vd,d),b_proj=(vh,d),a_proj=(vh,d),conv=(channels,1,kernel),
        a_log=(vh,),dt_bias=(vh,),norm=(vd,),out_proj=(d,vh*vd),**cs)
    fs=dict(q=(2*heads*dim,d),k=(kv*dim,d),v=(kv*dim,d),out=(d,heads*dim),q_norm=(dim,),k_norm=(dim,),**cs)
    for group,shapes,names,layer in [('linear',ls,dict(linear_names,**common),0),('full',fs,dict(full_names,**common),1)]:
        groups[group]={}
        for name,shape in shapes.items():
            key=group+'_'+name;weights[key]=rng.normal(0,.2,shape).astype(np.float32)
            suffixes[key]='language_model.layers.'+str(layer)+'.'+names[name];groups[group][name]=key
            if group=='linear' and name=='norm':weights[key]+=1
            if group=='linear' and name in ('a_log','dt_bias','norm'):f32.append(key)
    vs={'patch_embed.proj.weight':(16,3,2,2,2),'patch_embed.proj.bias':(16,),'pos_embed.weight':(9,16),
        'merger.norm.weight':(16,),'merger.norm.bias':(16,),'merger.linear_fc1.weight':(64,64),
        'merger.linear_fc1.bias':(64,),'merger.linear_fc2.weight':(d,64),'merger.linear_fc2.bias':(d,)}
    for layer in range(2):
        p='blocks.'+str(layer)
        for name in ('norm1','norm2'):vs[p+'.'+name+'.weight']=(16,);vs[p+'.'+name+'.bias']=(16,)
        for name,rows,cols in [('attn.qkv',48,16),('attn.proj',16,16),('mlp.linear_fc1',32,16),('mlp.linear_fc2',16,32)]:
            vs[p+'.'+name+'.weight']=(rows,cols);vs[p+'.'+name+'.bias']=(rows,)
    groups['vision']={}
    for name,shape in vs.items():
        key='vision_'+name;weights[key]=rng.normal(0,.12,shape).astype(np.float32)
        if '.norm' in name and name.endswith('.weight'):weights[key]+=1
        suffixes[key]='visual.'+name;groups['vision'][name]=key
    weights['embedding']=rng.normal(0,.2,(vocab,d)).astype(np.float32);suffixes['embedding']='language_model.embed_tokens.weight'
    weights['final_norm']=rng.normal(0,.2,(d,)).astype(np.float32);suffixes['final_norm']='language_model.norm.weight'
    with tempfile.TemporaryDirectory(prefix='garnet-metal-image-pipeline-') as tmp:
        tmp=Path(tmp);checkpoint=tmp/'weights.safetensors'
        decoded=write_checkpoint(checkpoint,weights,suffixes,'model',f32)
        lw,fw,vw=[{name:decoded[key] for name,key in groups[group].items()} for group in ('linear','full','vision')]
        requests=[]
        for request in range(2):
            patches=rng.normal(size=(batch,24,24)).astype(np.float32);patch_file=tmp/('image-'+str(request)+'.f32')
            patches.astype('<f4').tofile(patch_file);visual=reference(patches,vw,vc,4,6)
            conv=np.zeros((batch,channels,kernel),np.float32);state=np.zeros((batch,vh,kd,vd),np.float32)
            key=np.zeros((batch,kv,capacity,dim),np.float32);value=np.zeros_like(key);slots=np.zeros(batch,np.float32)
            steps=[]
            def step(tokens,visual_token,position,active):
                return dict(tokens=tokens,visual_token=visual_token,positions=[position]*batch,active=active)
            inputs=[step((np.arange(batch)+3).tolist(),None,[0,0,0],[1]*batch)]
            inputs.extend(step(None,i,[1,1+i//3,1+i%3],[1]*batch) for i in range(6))
            inputs.append(step((np.arange(batch)+8).tolist(),None,[4,4,4],[1]*batch))
            next_tokens=None
            for index in range(14):
                if index<8:item=inputs[index]
                else:
                    active=[1]*batch
                    if batch==2 and index in (10,11):active[1]=0
                    item=step(next_tokens.tolist(),None,[index-3]*3,active)
                active=np.asarray(item['active'],np.float32);positions=np.asarray(item['positions'],np.float32)
                hidden=visual[:,item['visual_token']] if item['visual_token'] is not None else decoded['embedding'][item['tokens']]
                hidden,conv,state=linear_block(hidden,conv,state,active,lw,kh,vh,kd,vd)
                hidden,key,value=attention_block(hidden,key,value,positions,slots,active,fw,heads,kv,dim,8,[2,1,1],10000000)
                logits=rms_norm(hidden,decoded['final_norm'])@decoded['embedding'].T;prediction=np.argmax(logits,axis=-1)
                state_files=[]
                for state_index,array in enumerate((conv,state,key,value)):
                    path=tmp/('state-'+str(request)+'-'+str(index)+'-'+str(state_index)+'.f32')
                    array.astype('<f4').tofile(path);state_files.append(dict(path=str(path),count=array.size))
                steps.append(dict(item,prediction=prediction.tolist(),state_files=state_files))
                next_tokens=np.where(active!=0,prediction,next_tokens) if index>=8 else prediction
                slots+=active
            visual_file=tmp/('visual-'+str(request)+'.f32');visual.astype('<f4').tofile(visual_file)
            requests.append(dict(patch_file=str(patch_file),visual_file=str(visual_file),visual_count=visual.size,steps=steps))
        fixture=tmp/'fixture.json';fixture.write_text(json.dumps(dict(checkpoint=str(checkpoint),config=c,
            vision_config=vc,batch=batch,capacity=capacity,requests=requests)))
        subprocess.run([sys.argv[1],str(Path(__file__).with_name('run_image_pipeline.py')),str(fixture),sys.argv[2],sys.argv[3]],check=True)
