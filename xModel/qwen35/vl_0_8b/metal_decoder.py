# SPDX-License-Identifier: Apache-2.0
"""Garnet Metal decoder: reuse explicit-state serving, project the head on GPU."""
import garnet,tensor,time
from .cpu_decoder import CPUDecoder
from .decoder_layer import T,norm,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer

class MetalDecoder(CPUDecoder):
    def __init__(self,checkpoint,config,batch=1,capacity=256):
        if not garnet.backend_info().startswith('Garnet Metal/MPS '):
            raise ValueError('MetalDecoder requires the native Garnet Metal package')
        CPUDecoder.__init__(self,checkpoint,config,batch,capacity)
        try:
            self.head.close()
            symbol=tensor.input('metal_final_hidden',shape=[batch,config['hidden_size']],dtype=tensor.float32)
            @T.fusion(name='qwen35_metal_tied_head',boundary='required')
            def head(hidden):
                return norm(hidden,'model.language_model.norm',config) * T.unary_op('qwen35_tied_head')
            self.head=garnet.compiled_graph();self.head.build(head(symbol),[symbol],checkpoint)
        except:
            self.close();raise

    def enable_prefill(self,chunk=4):
        if chunk not in (2,4):raise ValueError('Metal prefill chunk must be 2 or 4')
        if hasattr(self,'prefill_graphs') and getattr(self,'chunk',None)!=chunk:
            for graph in self.prefill_graphs.values():graph.close()
            self.prefill_graphs={}
        self.chunk=chunk
        if not hasattr(self,'prefill_graphs'):self.prefill_graphs={}

    def prefill_sequence(self,hidden_tokens,positions,predict=True):
        if self.poisoned:raise ValueError('reset after failed prefill')
        count=len(hidden_tokens);chunk=self.chunk;batch=self.batch;config=self.config;capacity=self.capacity
        if count<1 or any(slot+count>capacity for slot in self.slots):raise ValueError('invalid prefill length/capacity')
        if len(positions)!=batch or any(len(row)!=count or any(len(p)!=3 for p in row) for row in positions):raise ValueError('invalid prefill THW positions')
        from .prefill_layer import PrefillLayer
        width=config['hidden_size'];self.poisoned=True;self.prefill_compile_ms=0;self.prefill_inference_ms=0
        for idx,layer in enumerate(self.layers):
            kind=layer['kind'];shapes=[[batch,chunk,width],layer['shapes'][1],layer['shapes'][2],
                [batch,chunk,3],[batch,chunk],[batch,chunk]]
            symbols=[tensor.input('prefill_'+str(idx)+'_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
            @T.fusion(name='qwen35_metal_prefill',boundary='required')
            def forward(hidden,state1,state2,positions,slots,active):
                return PrefillLayer(hidden,state1,state2,positions,slots,active,config,idx,batch,chunk,capacity,kind)
            key=str(chunk)+'-'+str(idx)
            graph=self.prefill_graphs.get(key);next_hidden=[]
            if graph is None:
                graph=garnet.compiled_graph()
                try:graph.build(forward(*symbols),symbols,self.checkpoint)
                except:graph.close();raise
                self.prefill_graphs[key]=graph
                self.prefill_compile_ms+=graph.compile_ms
            for offset in range(0,count,chunk):
                values=hidden_tokens[offset:offset+chunk];length=len(values)
                while len(values)<chunk:values.append(self.io.zeros([batch,width]))
                hidden=self.io.stack_tokens(values)
                active=tensor.tensor([1 if i<length else 0 for b in range(batch) for i in range(chunk)],shape=[batch,chunk],dtype=tensor.float32)
                pos=[v for row in positions for i in range(chunk) for v in (row[offset+i] if i<length else [0,0,0])]
                position=tensor.tensor(pos,shape=[batch,chunk,3],dtype=tensor.float32)
                slots=tensor.tensor([slot+offset+min(i,length) for slot in self.slots for i in range(chunk)],shape=[batch,chunk],dtype=tensor.float32)
                packed=graph.run([hidden,layer['state1'],layer['state2'],position,slots,active]);self.prefill_inference_ms+=graph.last_inference_ms
                hidden=self.io.slice(packed,0,[batch,chunk,width])
                layer['state1']=self.io.slice(packed,chunk*width,layer['shapes'][1])
                size=1
                for d in layer['shapes'][1][1:]:size*=d
                layer['state2']=self.io.slice(packed,chunk*width+size,layer['shapes'][2])
                for i in range(length):next_hidden.append(self.io.visual_token(hidden,i))
            hidden_tokens=next_hidden
        prediction=None
        if predict:prediction=self.io.argmax(self.io.logits(self.head.run([hidden_tokens[-1]])))
        for b in range(batch):self.slots[b]+=count
        self.poisoned=False
        return prediction

    def _compile(self,checkpoint,config,idx,kind,shapes,capacity):
        symbols=[tensor.input('metal_layer_'+str(idx)+'_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
        if kind=='linear_attention':
            @T.fusion(name='qwen35_metal_linear_layer',boundary='required')
            def forward(hidden,conv,state,active):
                return Qwen35LinearDecoderLayer(hidden,conv,state,active,config,idx)
        else:
            @T.fusion(name='qwen35_metal_full_layer',boundary='required')
            def forward(hidden,key,value,positions,slots,active):
                return Qwen35FullDecoderLayer(hidden,key,value,positions,slots,active,config,idx,capacity)
        graph=garnet.compiled_graph()
        try:graph.build(forward(*symbols),symbols,checkpoint)
        except:graph.close();raise
        return graph

    def close(self):
        if hasattr(self,'prefill_graphs'):
            for graph in self.prefill_graphs.values():graph.close()
            self.prefill_graphs={}
        CPUDecoder.close(self)
        self.io.close()
