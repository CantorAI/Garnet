import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
with open(sys.argv[1]) as stream:f=json.load(stream)
io=garnet.tensor_io()
for case in f['cases']:
    batch=case['batch'];symbol=tensor.input('head_hidden',shape=[batch,f['width']],dtype=tensor.float32)
    graphs=[]
    try:
        for bf16 in (False,True):
            @T.fusion(name='head_storage_reference',boundary='required')
            def forward(hidden):
                normalized=hidden * T.unary_op('qwen35_rms_norm' if f['vl'] else 'qwen_text_norm',weight_name=f['norm'],eps=f['eps'])
                return normalized * T.unary_op('qwen_text_head',weight_name=f['weight'],bf16_weight=1 if bf16 else 0)
            graph=garnet.compiled_graph();graphs.append(graph)
            if bf16:graph.build_shared(forward(symbol),[symbol],f['checkpoint'],graphs[0])
            else:graph.build(forward(symbol),[symbol],f['checkpoint'])
        hidden=io.read_f32(case['hidden'],[batch,f['width']])
        old=graphs[0].run([hidden]);graphs[0].close();new=graphs[1].run([hidden])
        assert io.argmax(old)==io.argmax(new),'BF16 head changed greedy prediction'
        wanted=old.tolist();actual=new.tolist();width=len(wanted)//batch;worst_l2=0;worst_max=0
        for row in range(batch):
            reference=wanted[row*width:(row+1)*width];value=actual[row*width:(row+1)*width]
            differences=[x-y for x,y in zip(value,reference)]
            assert all(math.isfinite(x) for x in value),'Nonfinite BF16 vocabulary logits'
            l2=math.sqrt(sum(x*x for x in differences))/(5e-6+5e-5*math.sqrt(sum(y*y for y in reference)))
            peak=max(abs(x) for x in differences)/(5e-6+5e-5*max(abs(y) for y in reference))
            assert l2<=1 and peak<=1,'BF16 vocabulary logits exceeded vector norm tolerance'
            worst_l2=max(worst_l2,l2);worst_max=max(worst_max,peak)
        print('BF16 full vocabulary PASS',dict(batch=batch,checkpoint=f['checkpoint'],worst_l2_ratio=worst_l2,
            worst_max_ratio=worst_max,predictions=io.argmax(new),fp32_ms=graphs[0].last_inference_ms,bf16_ms=graphs[1].last_inference_ms),flush=True)
    finally:
        for graph in graphs:graph.close()
