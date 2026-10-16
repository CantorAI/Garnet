"""Capture norm, key projection and key-cache stages without changing kernels."""
import sys,json
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen.metal_dense.layers import T,norm,reshape,attention
from xModel.qwen35.vl_0_8b.full_attention import unpack
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];chunk=f['chunk'];width=c['hidden_size'];capacity=f['capacity'];prefix='model.layers.0'
flat=tensor.input('flat_hidden',shape=[batch*chunk,width],dtype=tensor.float32)
@T.fusion(name='dense_norm_stage',boundary='required')
def norm_stage(hidden):return norm(hidden,prefix+'.input_layernorm',c)
@T.fusion(name='dense_projection_stage',boundary='required')
def projection_stage(hidden):
    return norm(hidden,prefix+'.input_layernorm',c)*T.unary_op('qwen_text_head',weight_name=prefix+'.self_attn.k_proj.weight',bf16_weight=0)
normalized=norm_stage(flat);projected=projection_stage(flat)
graphs=[];results={}
try:
    data=garnet.tensor_io().read_f32(f['flat_input'],[batch*chunk,width])
    for label,output in [('normalized',normalized),('projected',projected)]:
        graph=garnet.compiled_graph();graphs.append(graph)
        graph.build_bf16(output,[flat],f['checkpoint'])
        results[label]=graph.run([data]).tolist()
    symbols=[tensor.input('key_stage_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(f['shapes'])]
    @T.fusion(name='dense_key_stage',boundary='required')
    def forward(hidden,key,value,positions,slots,active):
        hidden=hidden*T.binary_op('qwen35_mask_chunk')*active
        x=reshape(norm(reshape(hidden,[batch*chunk,width]),prefix+'.input_layernorm',c),[batch,chunk,width])
        packed=attention(x,key,value,positions,slots,active,c,0,chunk=True)
        cfg=dict(c);cfg['hidden_size']=chunk*width
        return unpack(packed,cfg,capacity)[1]
    graph=garnet.compiled_graph();graphs.append(graph)
    graph.build_bf16(forward(*symbols),symbols,f['checkpoint'])
    inputs=[garnet.tensor_io().read_f32(path,s) for path,s in zip(f['inputs'],f['shapes'])]
    results['key_cache']=graph.run(inputs).tolist()
    with open(f['output'],'w') as stream:stream.write(json.dumps(results))
finally:
    for graph in graphs:graph.close()
    garnet.shutdown()
