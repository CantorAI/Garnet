# SPDX-License-Identifier: Apache-2.0
"""Experimental prefix commits: replay small delta cores, no weight GEMMs.

Requires an explicitly retained projection payload in the target layer output.
All accepted hidden rows came from authoritative target verification. Partial
commits rerun only ordered convolution/delta recurrence from the saved initial
state; attention restores unaccepted KV suffixes from the existing backup.
This avoids per-token copies of the large recurrent states. No performance or
correctness acceptance until the unchanged native verifier/session tests pass.
"""
import json
import garnet
import tensor
from .decoder_layer import T
from .cuda_verify import CUDATargetVerifier

class CUDADeltaReplayVerifier(CUDATargetVerifier):
    def __init__(self,decoder,checkpoint,max_tokens=4):
        if not getattr(decoder,'retain_linear_inputs',False) or not decoder.fused_delta:
            raise ValueError('Delta replay requires retained target projections and fused delta')
        self.replay_graphs=[];self.cache_spans=None
        super().__init__(decoder,checkpoint,max_tokens)
        try:
            self.cache_spans=self.io.zeros([decoder.batch,2])
            for index,layer in enumerate(decoder.layers):
                if layer['kind']!='linear_attention':continue
                symbols=[tensor.input('replay_'+str(index)+'_'+str(i),shape=shape,dtype=tensor.float32)
                    for i,shape in enumerate([layer['packed_shape'],layer['shapes'][1],layer['shapes'][2],[decoder.batch,decoder.chunk_size]])]
                forward=self._forward(index)
                graph=garnet.compiled_graph()
                try:graph.build(forward(*symbols),symbols,checkpoint);graph.enable_cuda_graph(True)
                except:graph.close();raise
                self.replay_graphs.append((index,graph))
        except:self.close();raise

    def _forward(self,index):
        config=self.decoder.config;size=self.decoder.chunk_size
        @T.fusion(name='qwen35_delta_prefix_replay',boundary='required')
        def replay(packed,history,state,active):
            bound=packed*T.binary_op('qwen35_conv_bind_state')*history
            bound=bound*T.binary_op('qwen35_delta_bind_state')*state
            bound=bound*T.binary_op('qwen35_conv_bind_active')*active
            return bound*T.unary_op('qwen35_linear_replay_chunk',layer_prefix='model.language_model.layers.'+str(index),
                hidden_dim=config['hidden_size'],tokens=size,key_heads=config['linear_num_key_heads'],value_heads=config['linear_num_value_heads'],
                key_dim=config['linear_key_head_dim'],value_dim=config['linear_value_head_dim'],eps=config['rms_norm_eps'],register_delta_128=int(getattr(self.decoder,'register_delta_128',False)))
        return replay

    def commit_prefix(self,counts):
        if not self.pending:raise ValueError('No pending verification to commit')
        decoder=self.decoder;b=decoder.batch
        if len(counts)!=b or any(not isinstance(n,int) or n<0 or n>limit for n,limit in zip(counts,self.submitted)):
            raise ValueError('Accepted prefix exceeds submitted tokens')
        if counts==self.submitted:
            self.pending=False
            return [self.predictions[row*self.max_tokens+n-1] if n else None for row,n in enumerate(counts)]
        flags=[];slots=[]
        for row in range(b):
            slot=self.saved_slots[row]
            for t in range(decoder.chunk_size):
                active=int(t<counts[row]);flags.append(active);slots.append(slot);slot+=active
        try:
            self.io.update(decoder.active,flags);self.io.update(decoder.slot_tensor,slots)
            spans=[]
            for row in range(b):spans.extend([self.saved_slots[row],counts[row]])
            self.io.update(self.cache_spans,spans)
            for index,graph in self.replay_graphs:
                layer=decoder.layers[index]
                packed=graph.run_device([layer['last_packed'],self.backups[5+2*index],self.backups[6+2*index],decoder.active])
                self.io.slice_into(packed,0,layer['state1']);self.io.slice_into(packed,layer['state1_width'],layer['state2'])
            for index,layer in enumerate(decoder.layers):
                if layer['kind']!='full_attention':continue
                self.io.restore_kv_suffix(layer['state1'],self.backups[5+2*index],self.cache_spans)
                self.io.restore_kv_suffix(layer['state2'],self.backups[6+2*index],self.cache_spans)
            self.io.mask_chunk(decoder.hidden,decoder.active)
            self.io.copy_into(self.backups[1],decoder.last_hidden)
            self.io.copy_last_chunk(decoder.hidden,decoder.active,decoder.last_hidden)
            decoder.slots=[self.saved_slots[row]+counts[row] for row in range(b)];decoder.poisoned=False
            self._read();self.pending=False
            return [self.predictions[row*self.max_tokens+n-1] if n else None for row,n in enumerate(counts)]
        except:self.rollback();raise

    def memory_info(self):
        result=super().memory_info();graphs=[json.loads(graph.stats_json()) for _,graph in self.replay_graphs]
        spans=self.io.tensor_bytes(self.cache_spans) if self.cache_spans is not None else 0
        extra=sum(g['device_output_bytes']+g['trt_allocator_live_bytes'] for g in graphs) if all(g['allocator_tracking_enabled'] for g in graphs) else None
        result.update({'delta_replay_graphs':graphs,'cache_span_bytes':spans,'partial_commit_method':'replay_delta_restore_kv'})
        result['owned_helper_bytes']=result['owned_helper_bytes']+spans+extra if extra is not None and result['owned_helper_bytes'] is not None else None
        return result

    def close(self):
        try:super().close()
        finally:
            for _,graph in self.replay_graphs:graph.close()
            self.replay_graphs=[];self.cache_spans=None
