# SPDX-License-Identifier: Apache-2.0
"""Garnet Metal decoder: reuse explicit-state serving, project the head on GPU."""
import garnet,tensor
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
        raise ValueError('Metal chunked prefill is not implemented; use sequential token steps')

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
        CPUDecoder.close(self)
        self.io.close()
