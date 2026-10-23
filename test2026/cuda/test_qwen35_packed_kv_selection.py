"""Independent scalar span selection oracle, including signed-zero copies."""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
checkpoint,output=sys.argv[1:3];records=[]
def bits(values):return struct.pack('<'+str(len(values))+'f',*values)
def read(value):return bits(garnet.tensor_to_cpu(value).tolist())
for b,kv,capacity,dim in ((1,2,512,256),(2,1,32,8),(4,4,32,256),(8,4,32,256)):
    shape=[b,kv,capacity,dim];n=b*kv*capacity*dim
    current=[(-0.0 if i%31==0 else (i%31-15)*0.125) for i in range(n)]
    backup=[(0.0 if i%31==0 else (17-i%37)*0.25) for i in range(n)]
    io=garnet.cuda_tensor_io();io.open(checkpoint);graph=garnet.compiled_graph()
    try:
        x=io.zeros(shape);old=io.zeros(shape);spans=io.zeros([b,2]);io.update(x,current);io.update(old,backup)
        symbols=[tensor.input('kv_select_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate((shape,shape,[b,2]))]
        @T.fusion(name='qwen35_kv_selection_oracle',boundary='required')
        def forward(value,saved,ranges):
            value=value*T.binary_op('qwen35_cache_bind_backup')*saved
            value=value*T.binary_op('qwen35_cache_bind_spans')*ranges
            return value*T.unary_op('qwen35_restore_kv_prefix')
        graph.build(forward(*symbols),symbols,checkpoint);graph.enable_cuda_graph(True)
        for trial in range(6):
            ranges=[]
            for row in range(b):
                start=(trial*7+row*3)%capacity
                count=(0,1,2,capacity-start)[(trial+row)%4];count=min(count,capacity-start)
                ranges.extend([start,count])
            io.update(spans,ranges);actual=graph.run_device([x,old,spans]);expected=[]
            for row in range(b):
                start,count=ranges[2*row:2*row+2]
                for head in range(kv):
                    for slot in range(capacity):
                        offset=((row*kv+head)*capacity+slot)*dim
                        source=current if start<=slot<start+count else backup
                        expected.extend(source[offset:offset+dim])
            assert read(actual)==bits(expected),('Scalar prefix selection differs',b,trial)
            assert read(x)==bits(current) and read(old)==bits(backup),'Selection modified inputs'
        stats=json.loads(graph.stats_json());assert stats['graph_captures'] and stats['graph_replays'] and not stats['graph_capture_failures']
        records.append({'shape':shape,'six_span_patterns_bitwise_equal':True,'input_bytes_unchanged':True,'stats':stats})
    finally:graph.close();io.close()
with open(output,'w') as stream:json.dump({'all_passed':True,'records':records,'scope':'Independent scalar selection oracle; no model quality or speed acceptance'},stream,indent=2)
print('QWEN35_PACKED_KV_SELECTION_COMPLETE',len(records),flush=True)
