# SPDX-License-Identifier: Apache-2.0
"""Experimental prefill/decode profiles sharing one BF16 projection pool.

The caller owns one checkpoint IO. Both graph sets borrow its readonly matrix
views and share the exact state and last-hidden tensors. Only activations,
metadata, small constants and graph workspaces differ between profiles.
"""
import garnet
from .cuda_decoder import CUDADecoder

class CUDAHybridDecoder:
    def __init__(self,checkpoint,config,batch=1,capacity=512,prefill_chunk=64,decode_chunk=4,projection_workspace_bytes=0,retain_linear_inputs=False,register_delta_128=False):
        if decode_chunk not in (4,8,16) or prefill_chunk not in (32,64,128) or prefill_chunk<=decode_chunk:
            raise ValueError('Hybrid profiles require decode4/8/16 and larger prefill32/64/128')
        self.prefill_chunk_size=prefill_chunk;self.decode_chunk_size=decode_chunk;self.prefill_mode=False
        self.io=garnet.cuda_tensor_io();self.io.open(checkpoint);self.decode=None;self.wide=None
        try:
            self.decode=CUDADecoder(checkpoint,config,batch,capacity,decode_chunk,True,True,projection_workspace_bytes,retain_linear_inputs,
                external_weights=True,shared_io=self.io,register_delta_128=register_delta_128)
            self.wide=CUDADecoder(checkpoint,config,batch,capacity,prefill_chunk,True,True,projection_workspace_bytes,False,
                external_weights=True,shared_io=self.io,shared_decoder=self.decode,register_delta_128=register_delta_128)
        except:self.close();raise

    def _current(self):
        current=self.wide if self.prefill_mode else self.decode
        if current is None:raise ValueError('Hybrid decoder is closed')
        return current

    def __getattr__(self,name):
        current=self.__dict__.get('wide' if self.__dict__.get('prefill_mode',False) else 'decode')
        if current is None:raise AttributeError(name)
        return getattr(current,name)

    @property
    def slots(self):return self._current().slots
    @slots.setter
    def slots(self,value):self._current().slots=list(value)
    @property
    def poisoned(self):return self._current().poisoned
    @poisoned.setter
    def poisoned(self,value):self._current().poisoned=value
    @property
    def chunk_size(self):return self._current().chunk_size
    @property
    def resident_controls(self):return getattr(self.decode,'resident_controls',False)
    @resident_controls.setter
    def resident_controls(self,value):
        self.decode.resident_controls=value;self.wide.resident_controls=value

    def reset(self):
        if self.decode is None or self.wide is None:raise ValueError('Hybrid decoder is closed')
        self.prefill_mode=False;self.decode.reset()
        # State and last-hidden are already shared/reset. Only wide request
        # buffers and logical slots need clearing, without a second state pass.
        self.wide.slots=[0]*self.decode.batch;self.wide.poisoned=False
        for value in (self.wide.hidden,self.wide.active,self.wide.position,self.wide.slot_tensor):self.io.zero(value)

    def enable_cuda_graph(self,enabled):
        self.decode.enable_cuda_graph(enabled);self.wide.enable_cuda_graph(enabled)

    def chunk_hidden(self,hidden,active,positions,predict=True):return self._current().chunk_hidden(hidden,active,positions,predict)
    def step(self,tokens,active=None,positions=None,predict=True):return self._current().step(tokens,active,positions,predict)
    def step_hidden(self,hidden,active=None,positions=None,predict=True):return self._current().step_hidden(hidden,active,positions,predict)
    def chunk_tokens(self,tokens,active,positions,predict=True):return self._current().chunk_tokens(tokens,active,positions,predict)

    def prefill_sequence(self,hidden,positions,lengths=None):
        if self.prefill_mode or self.poisoned:raise ValueError('Reset a failed or active hybrid prefill')
        count=len(hidden);b=self.decode.batch
        if count<1 or len(positions)!=b or any(len(row)!=count or any(len(p)!=3 for p in row) for row in positions):raise ValueError('Invalid hybrid prefill shape')
        if lengths is None:lengths=[count]*b
        if len(lengths)!=b or any(n<1 or n>count for n in lengths) or any(self.decode.slots[row]+lengths[row]>self.decode.capacity for row in range(b)):
            raise ValueError('Hybrid prefill exceeds profile')
        self.wide.slots=list(self.decode.slots);self.prefill_mode=True
        try:
            for start in range(0,count,self.prefill_chunk_size):
                end=min(start+self.prefill_chunk_size,count)
                # Dispatch through this object so an MTP observer sees each
                # authoritative wide target chunk before its output is reused.
                prediction=self.chunk_hidden(hidden[start:end],[[int(t<lengths[row]) for row in range(b)] for t in range(start,end)],
                    [row[start:end] for row in positions],end==count)
            return prediction
        except:
            self.decode.poisoned=True;self.wide.poisoned=True;raise
        finally:
            self.decode.slots=list(self.wide.slots);self.prefill_mode=False

    def memory_info(self):
        decode=self.decode.memory_info();wide=self.wide.memory_info()
        shared_io=decode['embedding_head_shared_bytes']+decode['head_input_bytes']+decode['head_output_bytes']+decode['proposal_io_bytes']+decode['projection_weight_pool_bytes']
        last=self.io.tensor_bytes(self.decode.last_hidden)
        owned=decode['owned_decoder_bytes']+wide['owned_decoder_bytes']-shared_io-decode['state_bytes']-last if decode['owned_decoder_bytes'] is not None and wide['owned_decoder_bytes'] is not None else None
        result=dict(decode)
        result.update({'graphs':decode['graphs']+wide['graphs'],'request_buffer_bytes':decode['request_buffer_bytes']+wide['request_buffer_bytes']-last,
            'engine_output_bytes':decode['engine_output_bytes']+wide['engine_output_bytes'],
            'trt_runtime_allocator_live_bytes':decode['trt_runtime_allocator_live_bytes']+wide['trt_runtime_allocator_live_bytes'] if decode['trt_runtime_allocator_live_bytes'] is not None and wide['trt_runtime_allocator_live_bytes'] is not None else None,
            'owned_decoder_bytes':owned,'shared_profile_metadata':{'prefill_chunk':self.prefill_chunk_size,'decode_chunk':self.decode_chunk_size,
                'one_projection_pool':True,'one_state_set':True,'shared_last_hidden_bytes':last,'duplicate_ownership_subtracted_bytes':shared_io+decode['state_bytes']+last},
            'scope':'Two bounded graph sets sharing one readonly projection pool, table, state and last-hidden. Activation/metadata/small-constant/runtime buffers counted separately; not process peak.'})
        return result

    def close(self):
        try:
            if self.wide is not None:self.wide.close()
        finally:
            try:
                if self.decode is not None:self.decode.close()
            finally:
                self.wide=None;self.decode=None;self.io.close();self.prefill_mode=False
