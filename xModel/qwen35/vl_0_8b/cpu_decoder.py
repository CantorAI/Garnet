# SPDX-License-Identifier: Apache-2.0
"""Fixed-batch Garnet CPU decoder with explicit per-request recurrent/KV state.

Image embeddings can enter through step_hidden; this file does not implement
the vision encoder or claim image inference before that integration executes.
"""
import garnet
import tensor
import time
from .decoder_layer import T,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer,norm

class CPUDecoder:
    def __init__(self,checkpoint,config,batch=1,capacity=256):
        if batch not in (1,2) or capacity<1:raise ValueError('fixed batch 1/2 and positive capacity required')
        self.config=config;self.batch=batch;self.capacity=capacity
        self.io=garnet.tensor_io();self.io.open(checkpoint)
        self.layers=[];self.head=None
        try:
            for idx,kind in enumerate(config['layer_types']):
                if kind=='linear_attention':
                    channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
                    shapes=[[batch,config['hidden_size']],[batch,channels,config['linear_conv_kernel_dim']],
                        [batch,config['linear_num_value_heads'],config['linear_key_head_dim'],config['linear_value_head_dim']],[batch]]
                elif kind=='full_attention':
                    cache=[batch,config['num_key_value_heads'],capacity,config['head_dim']]
                    shapes=[[batch,config['hidden_size']],cache,cache,[batch,3],[batch],[batch]]
                else:raise ValueError('unsupported decoder layer '+kind)
                graph=self._compile(checkpoint,config,idx,kind,shapes,capacity)
                self.layers.append(dict(graph=graph,kind=kind,shapes=shapes))
                print('compiled Garnet decoder layer',idx,kind,'ms',graph.compile_ms,flush=True)
            symbol=tensor.input('final_hidden',shape=[batch,config['hidden_size']],dtype=tensor.float32)
            @T.fusion(name='qwen35_cpu_tied_head',boundary='required')
            def head(hidden):
                return norm(hidden,'model.language_model.norm',config)
            self.head=garnet.compiled_graph();self.head.build(head(symbol),[symbol],checkpoint)
            self.reset()
        except:
            self.close();raise

    def _compile(self,checkpoint,config,idx,kind,shapes,capacity):
        symbols=[tensor.input('layer_'+str(idx)+'_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
        @T.fusion(name='qwen35_cpu_decoder_layer',boundary='required')
        def forward(*args):
            if kind=='linear_attention':return Qwen35LinearDecoderLayer(*args,config,idx)
            return Qwen35FullDecoderLayer(*args,config,idx,capacity)
        graph=garnet.compiled_graph()
        try:graph.build(forward(*symbols),symbols,checkpoint)
        except:graph.close();raise
        return graph

    def reset(self):
        self.poisoned=False
        self.slots=[0 for i in range(self.batch)]
        for layer in self.layers:
            layer['state1']=self.io.zeros(layer['shapes'][1]);layer['state2']=self.io.zeros(layer['shapes'][2])

    def step(self,token_ids,active=None,positions=None,predict=True):
        if len(token_ids)!=self.batch:raise ValueError('token batch mismatch')
        return self.step_hidden(self.io.embedding(token_ids),active,positions,predict)

    def step_hidden(self,hidden,active=None,positions=None,predict=True):
        if self.poisoned:raise ValueError('reset request state after a failed decoder step')
        if active is None:active=[1 for i in range(self.batch)]
        if len(active)!=self.batch or any(a not in (0,1) for a in active):raise ValueError('invalid active rows')
        if any(active[b] and self.slots[b]>=self.capacity for b in range(self.batch)):raise ValueError('context capacity exhausted')
        if positions is None:positions=[[self.slots[b]]*3 for b in range(self.batch)]
        if len(positions)!=self.batch or any(len(p)!=3 for p in positions):raise ValueError('positions must be batch x THW')
        flag=tensor.tensor(active,shape=[self.batch],dtype=tensor.float32)
        position=tensor.tensor([v for row in positions for v in row],shape=[self.batch,3],dtype=tensor.float32)
        slots=tensor.tensor(self.slots,shape=[self.batch],dtype=tensor.float32)
        self.last_layer_ms=[]
        self.poisoned=True
        for layer in self.layers:
            if layer['kind']=='linear_attention':inputs=[hidden,layer['state1'],layer['state2'],flag]
            else:inputs=[hidden,layer['state1'],layer['state2'],position,slots,flag]
            packed=layer['graph'].run(inputs);self.last_layer_ms.append(layer['graph'].last_inference_ms)
            shapes=layer['shapes'];width=self.config['hidden_size']
            hidden=self.io.slice(packed,0,shapes[0])
            layer['state1']=self.io.slice(packed,width,shapes[1])
            size=1
            for d in shapes[1][1:]:size*=d
            layer['state2']=self.io.slice(packed,width+size,shapes[2])
        self.last_head_ms=0
        if predict:
            normalized=self.head.run([hidden]);head_start=time.time();logits=self.io.logits(normalized)
            self.last_head_ms=self.head.last_inference_ms+(time.time()-head_start)*1000
        prediction=self.io.argmax(logits) if predict else None
        for b in range(self.batch):self.slots[b]+=active[b]
        self.poisoned=False
        return prediction

    def close(self):
        if self.head is not None:self.head.close()
        for layer in self.layers:layer['graph'].close()
