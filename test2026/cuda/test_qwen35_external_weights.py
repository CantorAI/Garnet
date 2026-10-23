"""Unwired pool prototype: same BF16 matrices as inputs to multiple engines.

Tests native constant/external equivalence, input lifetime, graph replay and
allocation ownership. This is not independent numerical or model-speed proof.
"""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
fixture=json.load(open(sys.argv[1]));output=sys.argv[2];checkpoint=fixture['checkpoint']
group_mlp=len(sys.argv)>3 and sys.argv[3]=='1'
prefix='model.language_model.layers.0.mlp';names=[prefix+'.'+kind+'.weight' for kind in ('gate_proj','up_proj','down_proj')]
io=garnet.cuda_tensor_io();io.open(checkpoint);graphs=[];records=[]
def cpu_bits(t):
    values=garnet.tensor_to_cpu(t).tolist();return struct.pack('<'+str(len(values))+'f',*values)
def build(rows,external,invalid=None):
    h=fixture['config']['hidden_size'];symbol=tensor.input('external_mlp_input',shape=[rows,h],dtype=tensor.float32);symbols=[symbol];bindings={}
    if external:
        for i,name in enumerate(names):
            info=json.loads(io.weight_info(name));shape=list(info['shape'])
            if invalid=='shape' and i==0:shape[0]+=1
            dtype=tensor.float32 if invalid=='dtype' and i==0 else tensor.bfloat16
            weight=tensor.input('external_mlp_weight_'+str(i),shape=shape,dtype=dtype)
            symbols.append(weight);bindings[name]=weight
    @T.fusion(name='qwen35_external_weight_probe',boundary='required')
    def forward(hidden,*weights):
        # Refer to symbolic weights in attributes so capture records dependencies.
        attributes={'weight_prefix':prefix}
        if external:
            binding_names=list(names)
            if invalid=='missing':binding_names[0]='missing.weight'
            if invalid=='unused':binding_names[0]=binding_names[0].replace('layers.0.','layers.1.')
            attributes['external_weights']={binding_names[i]:weight for i,weight in enumerate(weights)}
        return hidden*T.unary_op('qwen35_mlp',**attributes)
    graph=garnet.compiled_graph()
    try:graph.build(forward(*symbols),symbols,checkpoint);graph.enable_cuda_graph(True)
    except:graph.close();raise
    graphs.append(graph);return graph
try:
    if group_mlp:io.group_weights(names[:2])
    weights=[io.weight(name) for name in names];pool=json.loads(io.memory_json())
    expected=sum(json.loads(io.weight_info(name))['bytes'] for name in names)
    assert pool['projection_weight_pool_bytes']==expected and pool['projection_weight_pool_entries']==3
    if group_mlp:assert pool['joined_projection_group_count']==1 and pool['joined_projection_member_count']==2
    for name in ('model.language_model.embed_tokens.weight','missing.weight','model.language_model.layers.0.input_layernorm.weight'):
        rejected=False
        try:io.weight(name)
        except BaseException:rejected=True
        assert rejected and json.loads(io.memory_json())['projection_weight_pool_bytes']==expected
    rejected=False
    try:io.copy_into(weights[0],weights[0])
    except BaseException:rejected=True
    assert rejected,'Pooled matrix view is writable'
    rejections=[]
    for invalid in ('shape','dtype','missing','unused'):
        rejected=False
        try:build(1,True,invalid)
        except BaseException as error:
            message=str(error);assert ('unused' in message if invalid=='unused' else 'External BF16 matrix input' in message),message
            rejected=True;rejections.append({'kind':invalid,'reason':message})
        assert rejected,('Invalid external binding accepted',invalid)
    for _ in range(3):
        repeated=[io.weight(name) for name in names]
        assert json.loads(io.memory_json())['projection_weight_pool_bytes']==expected
    for rows in (1,4,16):
        hidden=io.from_host([((i*17)%31-15)*.0125 for i in range(rows*fixture['config']['hidden_size'])],[rows,fixture['config']['hidden_size']])
        control=build(rows,False);external=build(rows,True)
        expected_bits=cpu_bits(control.run_device([hidden]));actual=external.run_device([hidden]+weights)
        assert cpu_bits(actual)==expected_bits,('External versus constants',rows)
        for repeat in range(3):assert cpu_bits(external.run_device([hidden]+weights))==expected_bits
        stats=json.loads(external.stats_json());assert stats['graph_replays']>0 and stats['graph_capture_failures']==0
        assert stats['external_bf16_weight_inputs']==3 and stats['external_bf16_weight_bytes']==expected
        control.close();assert cpu_bits(external.run_device([hidden]+weights))==expected_bits
        records.append({'rows':rows,'constant_external_bitwise_equal':True,'external_graph':stats,'pool_bytes':expected})
    for graph in graphs[:-1]:graph.close()
    assert json.loads(io.memory_json())['projection_weight_pool_bytes']==expected,'Graph close freed caller pool'
    io.close();assert json.loads(io.memory_json())['projection_weight_pool_bytes']==0
    # Caller-retained readonly views own their allocations after the pool closes.
    assert cpu_bits(graphs[-1].run_device([hidden]+weights))==expected_bits
    graphs[-1].close()
    result={'records':records,'owned_weight_pool_bytes':expected,'all_passed':True,'grouped_mlp_weights':group_mlp,'invalid_binding_rejections':rejections,
        'retained_views_survive_pool_close':True,
        'scope':'Standalone native constant/external equality and one caller pool, no independent math/full-model or speed acceptance'}
    with open(output,'w') as stream:json.dump(result,stream,indent=2)
    print('QWEN35_EXTERNAL_WEIGHTS_COMPLETE',len(records),flush=True)
finally:
    for graph in graphs:graph.close()
    io.close()
