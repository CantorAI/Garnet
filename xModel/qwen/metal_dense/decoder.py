# SPDX-License-Identifier: Apache-2.0
"""Native Metal dense Qwen text adapter, with explicit per-request KV state."""
import garnet,tensor
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
from .layers import T,DenseLayer,norm

class DenseMetalDecoder(MetalDecoder):
    def __init__(self,checkpoint,config,batch=1,capacity=256,bf16_head=True,resident=True,bf16_projections=True):
        self.session=None;self.bf16_projections=bf16_projections
        if not garnet.backend_info().startswith('Garnet Metal/MPS '):raise ValueError('Native Metal package required')
        if config.get('model_type') not in ('qwen3','qwen2'):raise ValueError('Only dense Qwen3/Qwen2 supported')
        if config.get('rope_scaling') is not None or config.get('use_sliding_window',False):raise ValueError('Scaled RoPE/sliding-window profiles are not implemented')
        if config.get('hidden_act','silu')!='silu':raise ValueError('Only SwiGLU dense MLP supported')
        if config['model_type']=='qwen3' and config.get('attention_bias',False):raise ValueError('Qwen3 projection bias is not implemented')
        if batch not in (1,2) or capacity<1:raise ValueError('Fixed batch 1/2 and positive capacity required')
        self.config=dict(config);self.config['head_dim']=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
        config=self.config;self.batch=batch;self.capacity=capacity;self.checkpoint=checkpoint
        self.layers=[];self.head=None;self.io=garnet.tensor_io()
        try:
            self.io.open(checkpoint)
            for idx in range(config['num_hidden_layers']):
                cache=[batch,config['num_key_value_heads'],capacity,config['head_dim']]
                shapes=[[batch,config['hidden_size']],cache,cache,[batch,3],[batch],[batch]]
                symbols=[tensor.input('dense_'+str(idx)+'_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
                @T.fusion(name='qwen_dense_metal_layer',boundary='required')
                def forward(hidden,key,value,positions,slots,active):
                    return DenseLayer(hidden,key,value,positions,slots,active,config,idx,capacity)
                graph=garnet.compiled_graph()
                try:self._build_graph(graph,forward(*symbols),symbols,checkpoint)
                except:graph.close();raise
                self.layers.append(dict(graph=graph,kind='full_attention',shapes=shapes))
            symbol=tensor.input('dense_final_hidden',shape=[batch,config['hidden_size']],dtype=tensor.float32)
            head_weight='model.embed_tokens.weight' if config.get('tie_word_embeddings',False) else 'lm_head.weight'
            @T.fusion(name='qwen_dense_metal_head',boundary='required')
            def head(hidden):
                return norm(hidden,'model.norm',config) * T.unary_op('qwen_text_head',weight_name=head_weight,bf16_weight=1 if bf16_head else 0)
            self.head=garnet.compiled_graph();self.head.build(head(symbol),[symbol],checkpoint)
            self.reset()
            if resident:self.enable_resident()
        except:self.close();raise

    def _prefill_layer(self,hidden,state1,state2,positions,slots,active,config,idx,batch,chunk,capacity,kind):
        from .layers import DensePrefillLayer
        return DensePrefillLayer(hidden,state1,state2,positions,slots,active,config,idx,batch,chunk,capacity,kind)
