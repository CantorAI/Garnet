import garnet
import tensor
from xModel.qwen35.vl_0_8b.gated_delta import Qwen35GatedDeltaStep
for batch in (1,2):
    shapes=[[batch,2,7],[batch,2,7],[batch,2,5],[batch,2],[batch,2],[batch,2,7,5],[batch]]
    inputs=[tensor.input("graph_delta_"+str(batch)+"_"+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
    graph=Qwen35GatedDeltaStep(*inputs)
    print("captured graph type",type(graph),"nodes",len(graph.inspect()),flush=True)
    assert garnet.run_graph(graph,inputs,batch)
    print("captured xModel to Garnet OpenVINO batch",batch,"PASS",flush=True)
