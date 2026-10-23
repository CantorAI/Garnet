"""Variadic graph keyword dependencies and row ordering versus a CPU oracle."""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet,tensor
from xModel.qwen35.vl_0_8b.cuda_packed_program import flat_concat
from xModel.qwen35.vl_0_8b.decoder_layer import T
checkpoint,output=sys.argv[1:3];records=[]
def bits(values):return struct.pack('<'+str(len(values))+'f',*values)
for b,count in ((1,2),(1,69),(8,69),(8,129)):
    io=garnet.cuda_tensor_io();io.open(checkpoint);graph=garnet.compiled_graph()
    try:
        widths=[1+(i*7)%37 for i in range(count)]
        symbols=[tensor.input('flat_part_'+str(i),shape=[b,w],dtype=tensor.float32) for i,w in enumerate(widths)]
        values=[io.zeros([b,w]) for w in widths]
        @T.fusion(name='flat_concat_oracle',boundary='required')
        def forward(*parts):return flat_concat(list(parts))
        graph.build(forward(*symbols),symbols,checkpoint);graph.enable_cuda_graph(True)
        for trial in range(3):
            host=[]
            for i,(value,width) in enumerate(zip(values,widths)):
                data=[(-0.0 if j%17==0 else (i*19+j+trial)*0.125) for j in range(b*width)];host.append(data);io.update(value,data)
            actual=garnet.tensor_to_cpu(graph.run_device(values)).tolist()
            expected=[number for row in range(b) for data,width in zip(host,widths) for number in data[row*width:(row+1)*width]]
            assert bits(actual)==bits(expected),('Flat concatenation bytes differ',b,count,trial)
        stats=json.loads(graph.stats_json());assert stats['graph_captures'] and stats['graph_replays'] and not stats['graph_capture_failures']
        records.append({'batch':b,'parts':count,'byte_exact_cpu_oracle':True,'stats':stats})
    finally:graph.close();io.close()
with open(output,'w') as stream:json.dump({'all_passed':True,'records':records,'scope':'Independent row ordering, signed-zero and changing-input graph replay; no model speed/quality claim'},stream,indent=2)
print('QWEN35_FLAT_CONCAT_COMPLETE',len(records),flush=True)
