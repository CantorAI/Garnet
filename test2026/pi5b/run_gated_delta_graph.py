import garnet
import tensor
from xModel.qwen35.vl_0_8b.gated_delta import Qwen35GatedDeltaStep
from xModel.qwen35.vl_0_8b.causal_conv import Qwen35CausalConvStep
from xModel.qwen35.vl_0_8b import linear_attention as linear

@linear.T.fusion(name="linear_capture_probe",boundary="required")
def capture_linear(hidden,conv_state,state,active,config):
    packed=linear.Qwen35LinearAttentionStep(hidden,conv_state,state,active,config,0)
    return linear.unpack(packed,config)
for batch in (1,2):
    shapes=[[batch,2,7],[batch,2,7],[batch,2,5],[batch,2],[batch,2],[batch,2,7,5],[batch]]
    inputs=[tensor.input("graph_delta_"+str(batch)+"_"+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
    graph=Qwen35GatedDeltaStep(*inputs)
    print("captured graph type",type(graph),"nodes",len(graph.inspect()),flush=True)
    assert garnet.run_graph(graph,inputs,batch)
    print("captured xModel to Garnet OpenVINO batch",batch,"PASS",flush=True)
    conv_shapes=[[batch,9],[9,4],[batch,9,4],[batch]]
    conv_inputs=[tensor.input("graph_conv_"+str(batch)+"_"+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(conv_shapes)]
    conv_graph=Qwen35CausalConvStep(*conv_inputs)
    assert garnet.run_conv(conv_graph,conv_inputs,batch)
    print("captured causal convolution batch",batch,"PASS",flush=True)
    config=dict(hidden_size=11,linear_num_key_heads=2,linear_num_value_heads=2,
        linear_key_head_dim=7,linear_value_head_dim=5,linear_conv_kernel_dim=4,rms_norm_eps=1e-6)
    linear_shapes=[[batch,11],[batch,38,4],[batch,2,7,5],[batch]]
    linear_inputs=[tensor.input("linear_"+str(batch)+"_"+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(linear_shapes)]
    nodes=capture_linear(*linear_inputs,config).inspect()
    seen=set()
    for node in nodes:
        assert all(dependency in seen for dependency in node['inputs'])
        seen.add(node['id'])
    assert len([node for node in nodes if node['provider']=='garnet'])==7
    print("linear attention xModel capture/extraction batch",batch,"PASS",flush=True)
