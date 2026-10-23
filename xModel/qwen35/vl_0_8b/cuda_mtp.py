# SPDX-License-Identifier: Apache-2.0
"""Experimental MTP program, used by the default-off verified session.

Caller supplies normalized target hidden rows, shifted next-token embeddings,
explicit KV caches/positions/slots/active controls and owns their lifetimes.
The tied embedding/head remains with the caller's existing CUDATensorIO.
"""
import garnet
import tensor
from .decoder_layer import T

@T.fusion(name='qwen35_cuda_mtp_chunk',role='decoder_layer',boundary='required')
def Qwen35MTPChunk(target,embedding,key,value,positions,slots,active,config,workspace_bytes=0):
    bound=target*T.binary_op('qwen35_mtp_bind_embedding')*embedding
    bound=bound*T.binary_op('qwen35_attention_bind_key')*key
    bound=bound*T.binary_op('qwen35_attention_bind_value')*value
    bound=bound*T.binary_op('qwen35_attention_bind_positions')*positions
    bound=bound*T.binary_op('qwen35_attention_bind_slots')*slots
    bound=bound*T.binary_op('qwen35_attention_bind_active')*active
    rope=config['rope_parameters']
    return bound*T.unary_op('qwen35_mtp_decoder_chunk',num_heads=config['num_attention_heads'],
        num_kv_heads=config['num_key_value_heads'],head_dim=config['head_dim'],
        rotary_dim=int(config['head_dim']*rope['partial_rotary_factor']),rope_theta=rope['rope_theta'],
        mrope_t=rope['mrope_section'][0],mrope_h=rope['mrope_section'][1],mrope_w=rope['mrope_section'][2],
        eps=config['rms_norm_eps'],projection_workspace_bytes=workspace_bytes)

class CUDAMTPProgram:
    def __init__(self,checkpoint,config,batch=1,capacity=512,chunk_size=16,projection_workspace_bytes=0):
        if batch not in (1,2,4,8) or not 32<=capacity<=8192 or chunk_size not in (1,4,8,16):
            raise ValueError('Unsupported MTP batch/capacity/chunk profile')
        if projection_workspace_bytes not in (0,16384):raise ValueError('MTP workspace must be0 or16384bytes')
        if config.get('mtp_num_hidden_layers',1)!=1:raise ValueError('MTP supports exactly one checkpoint layer')
        hidden=config['hidden_size'];cache=[batch,config['num_key_value_heads'],capacity,config['head_dim']]
        shapes=[[batch,chunk_size,hidden],[batch,chunk_size,hidden],cache,cache,
            [batch,chunk_size,3],[batch,chunk_size],[batch,chunk_size]]
        symbols=[tensor.input('cuda_mtp_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
        self.graph=garnet.compiled_graph()
        try:self.graph.build(Qwen35MTPChunk(*symbols,config,projection_workspace_bytes),symbols,checkpoint)
        except:self.graph.close();self.graph=None;raise
    def run(self,inputs):
        if self.graph is None:raise ValueError('MTP program is closed')
        if len(inputs)!=7:raise ValueError('MTP program requires seven input tensors')
        # Reused storage; consume/copy packed outputs before the next invocation.
        return self.graph.run_device(inputs)
    def close(self):
        if self.graph is not None:self.graph.close();self.graph=None
