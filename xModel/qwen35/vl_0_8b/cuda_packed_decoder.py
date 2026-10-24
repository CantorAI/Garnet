# SPDX-License-Identifier: Apache-2.0
"""Experimental whole-program decoder; opt-in integration follows validation.

By default both chunk profiles borrow one readonly projection pool and one
canonical state tensor. Opt-in mixed compressed inputs also share one pool.
Embedded projections require one shared chunk profile.
FP32 layer materialization remains explicit in the compiled program.
"""
import json,threading
import garnet
import tensor
from .decoder_layer import T,norm
from .cuda_decoder import CUDADecoder
from .cuda_packed_program import CUDAPackedDecoderProgram
from .joined_mlp_profile import validate_joined_mlp_profile

class CUDAPackedDecoder:
    def __init__(self,checkpoint,config,batch=1,capacity=512,prefill_chunk=64,decode_chunk=4,projection_workspace_bytes=0,retain_linear_inputs=False,register_delta_128=False,flat_concat=False,deferred_state_commit=False,shared_output_storage=False,join_decode_input_projections=False,embedded_projection_weights=False,external_compressed_weights=False,joined_mlp_projections=False):
        if decode_chunk not in (4,8,16) or prefill_chunk not in (4,8,16,32,64,128) or prefill_chunk<decode_chunk:
            raise ValueError('Packed decoder requires bounded prefill>=decode chunk')
        for size in (prefill_chunk,decode_chunk):validate_joined_mlp_profile(joined_mlp_projections,batch,capacity,size,external_compressed_weights,embedded_projection_weights,join_decode_input_projections)
        self.joined_mlp_projections=joined_mlp_projections
        self.config=config;self.batch=batch;self.capacity=capacity;self.checkpoint=str(checkpoint)
        self.embedded_projection_weights=bool(embedded_projection_weights)
        self.external_compressed_weights=bool(external_compressed_weights)
        if self.external_compressed_weights and (self.embedded_projection_weights or join_decode_input_projections):
            raise ValueError('Shared compressed projections exclude embedded/joined BF16 modes')
        if self.embedded_projection_weights and (prefill_chunk!=decode_chunk or join_decode_input_projections):
            raise ValueError('Embedded projections require one shared chunk profile and no BF16 joined pool')
        self.prefill_chunk_size=prefill_chunk;self.decode_chunk_size=decode_chunk;self.prefill_mode=False
        self.projection_workspace_bytes=projection_workspace_bytes;self.retain_linear_inputs=retain_linear_inputs
        self.register_delta_128=register_delta_128;self.flat_concat=flat_concat
        self.join_decode_input_projections=bool(join_decode_input_projections)
        if self.join_decode_input_projections and decode_chunk!=4:raise ValueError('Joined decode input projections require width4')
        self.deferred_state_commit=bool(deferred_state_commit)
        if self.deferred_state_commit and not retain_linear_inputs:raise ValueError('Deferred state requires retained projections and the packed replay verifier')
        self.state_transaction=None;self.transaction_thread=None;self.verification_chunk_allowed=False
        self.shared_output_storage=bool(shared_output_storage);self.output_pool=None
        self.fused_delta=True;self.fused_attention=True;self.packed_state_program=True
        self.programs={};self.requests={};self.head=None;self.state=None;self.last_hidden=None;self.last_packed=None
        self.io=garnet.cuda_tensor_io()
        try:
            self.io.open(checkpoint)
            for size in dict.fromkeys((decode_chunk,prefill_chunk)):
                program=CUDAPackedDecoderProgram(checkpoint,config,self.io,batch,capacity,size,retain_linear_inputs and size==decode_chunk,projection_workspace_bytes,register_delta_128,flat_concat,self.join_decode_input_projections and size==decode_chunk,self.embedded_projection_weights,self.external_compressed_weights,self.joined_mlp_projections)
                self.programs[size]=program
                if self.state is None:self.state=self.io.zeros([batch,program.state_width])
                self.requests[size]=[self.io.zeros(shape) for shape in ([batch,size,config['hidden_size']],[batch,size],[batch,size,3],[batch,size])]
            if self.shared_output_storage:
                width=max(p.output_width for p in self.programs.values())
                self.output_pool=self.io.zeros([batch,width])
                for p in self.programs.values():
                    p.graph.set_output_storage(self.io.storage_view(self.output_pool,[batch,p.output_width]))
            self.last_hidden=self.io.zeros([batch,config['hidden_size']])
            symbol=tensor.input('whole_final_hidden',shape=[batch,config['hidden_size']],dtype=tensor.float32)
            @T.fusion(name='qwen35_whole_final_norm',boundary='required')
            def normalize(value):return norm(value,'model.language_model.norm',config)
            self.head=garnet.compiled_graph();self.head.build(normalize(symbol),[symbol],checkpoint)
            self.reset()
        except:self.close();raise

    @property
    def chunk_size(self):return self.prefill_chunk_size if self.prefill_mode else self.decode_chunk_size
    @property
    def hidden(self):return self.requests[self.chunk_size][0]
    @property
    def active(self):return self.requests[self.chunk_size][1]
    @property
    def position(self):return self.requests[self.chunk_size][2]
    @property
    def slot_tensor(self):return self.requests[self.chunk_size][3]

    def verification_tensors(self):
        if self.prefill_mode:raise ValueError('Cannot verify during packed prefill')
        values=[self.hidden,self.last_hidden,self.active,self.position,self.slot_tensor]
        return values if self.deferred_state_commit else values+[self.state]

    def _claim_state_transaction(self,owner):
        if not self.deferred_state_commit or self.state_transaction is not None:raise ValueError('Canonical state is already borrowed or deferred verification is disabled')
        self.state_transaction=owner;self.transaction_thread=threading.get_ident();self.verification_chunk_allowed=True

    def _release_state_transaction(self,owner):
        if self.state_transaction is owner:
            self.state_transaction=None;self.transaction_thread=None;self.verification_chunk_allowed=False

    def reset(self):
        if self.state_transaction is not None:raise ValueError('Commit or roll back verification before resetting the decoder')
        if self.head is None:raise ValueError('Packed decoder is closed')
        self.prefill_mode=False;self.slots=[0]*self.batch;self.poisoned=False;self.last_packed=None
        self.io.zero(self.state);self.io.zero(self.last_hidden)
        for buffers in self.requests.values():
            for value in buffers:self.io.zero(value)

    def enable_cuda_graph(self,enabled):
        for program in self.programs.values():
            program.graph.enable_cuda_graph(enabled)
            program.graph.enable_serial_graph_reuse(bool(getattr(self,'serial_target_graph_reuse',False)))
        self.head.enable_cuda_graph(enabled)
        self.head.enable_serial_graph_reuse(bool(getattr(self,'serial_target_graph_reuse',False)))

    step=CUDADecoder.step
    step_hidden=CUDADecoder.step_hidden
    chunk_tokens=CUDADecoder.chunk_tokens

    def chunk_hidden(self,hidden_tokens,active_tokens,positions,predict=True,_token_ids=False):
        if self.head is None or self.poisoned:raise ValueError('Reset an open packed decoder before a chunk')
        deferred=self.state_transaction is not None
        if deferred and (not self.verification_chunk_allowed or self.transaction_thread!=threading.get_ident()):
            raise ValueError('Commit or roll back verification before using the decoder')
        count=len(hidden_tokens);size=self.chunk_size;b=self.batch
        if not 1<=count<=size or len(active_tokens)!=count or any(len(row)!=b or any(v not in (0,1) for v in row) for row in active_tokens):
            raise ValueError('Invalid packed decoder chunk controls')
        if len(positions)!=b or any(len(row)!=count or any(len(p)!=3 for p in row) for row in positions):raise ValueError('Invalid packed decoder positions')
        flags=[];slots=[];pos=[];next_slots=list(self.slots)
        for row in range(b):
            for t in range(size):
                active=active_tokens[t][row] if t<count else 0
                if active and next_slots[row]>=self.capacity:raise ValueError('Packed decoder context exhausted')
                flags.append(active);slots.append(next_slots[row]);next_slots[row]+=active
                pos.extend(positions[row][t] if t<count else [0,0,0])
        if deferred:self.verification_chunk_allowed=False
        self.poisoned=True
        if _token_ids:self.io.embedding_chunk(hidden_tokens,self.hidden)
        else:
            self.io.zero(self.hidden)
            for t,value in enumerate(hidden_tokens):self.io.write_chunk_token(value,self.hidden,t)
        if getattr(self,'resident_controls',False):self.io.update_request(self.active,self.position,self.slot_tensor,flags,pos,slots)
        else:
            self.io.update(self.active,flags);self.io.update(self.position,pos);self.io.update(self.slot_tensor,slots)
        program=self.programs[size]
        packed=program.run([self.hidden,self.state,self.active,self.position,self.slot_tensor])
        self.io.slice_into(packed,0,self.hidden)
        if not deferred:self.io.slice_into(packed,program.hidden_width,self.state)
        if size==self.decode_chunk_size:self.last_packed=packed
        self.io.copy_last_chunk(self.hidden,self.active,self.last_hidden)
        prediction=self.io.predict(self.head.run_device([self.last_hidden])) if predict else None
        self.slots=next_slots;self.last_layer_enqueue_ms=[program.graph.last_inference_ms];self.poisoned=False
        return prediction

    def prefill_sequence(self,hidden,positions,lengths=None):
        if self.state_transaction is not None:raise ValueError('Commit or roll back verification before prefill')
        if self.head is None or self.prefill_mode or self.poisoned:raise ValueError('Reset an open packed decoder before prefill')
        count=len(hidden);b=self.batch
        if count<1 or len(positions)!=b or any(len(row)!=count or any(len(p)!=3 for p in row) for row in positions):raise ValueError('Invalid packed prefill shape')
        if lengths is None:lengths=[count]*b
        if len(lengths)!=b or any(n<1 or n>count for n in lengths) or any(self.slots[row]+lengths[row]>self.capacity for row in range(b)):
            raise ValueError('Packed prefill exceeds profile')
        self.prefill_mode=True
        try:
            for start in range(0,count,self.prefill_chunk_size):
                end=min(start+self.prefill_chunk_size,count)
                prediction=self.chunk_hidden(hidden[start:end],[[int(t<lengths[row]) for row in range(b)] for t in range(start,end)],
                    [row[start:end] for row in positions],end==count)
            return prediction
        except:self.poisoned=True;raise
        finally:self.prefill_mode=False

    def memory_info(self):
        if self.head is None:raise ValueError('Packed decoder is closed')
        graphs=[program.memory_info()['graph'] for program in self.programs.values()]+[json.loads(self.head.stats_json())]
        info=json.loads(self.io.memory_json());tracked=all(g['allocator_tracking_enabled'] for g in graphs)
        runtime=sum(g['trt_allocator_live_bytes'] for g in graphs) if tracked else None
        states=self.io.tensor_bytes(self.state)
        requests=sum(self.io.tensor_bytes(value) for buffers in self.requests.values() for value in buffers)+self.io.tensor_bytes(self.last_hidden)
        pool_bytes=self.io.tensor_bytes(self.output_pool) if self.output_pool is not None else 0
        outputs=sum(g.get('owned_device_output_bytes',g['device_output_bytes']) for g in graphs)+pool_bytes
        proposal=sum(info.get(key,0) for key in ('candidate_workspace_bytes','candidate_index_bytes','indexed_head_buffers_bytes'))
        shared=info.get('embedding_table_bytes',info.get('embedding_head_shared_bytes',0))+sum(info.get(key,0) for key in ('untied_head_bytes','native_head_workspace_bytes','head_input_bytes','head_output_bytes','projection_weight_pool_bytes','compressed_projection_weight_bytes','compressed_projection_scale_bytes'))
        return {'graphs':graphs,'state_bytes':states,'request_buffer_bytes':requests,'engine_output_bytes':outputs,'shared_output_pool_bytes':pool_bytes,
            'embedding_head_shared_bytes':info['embedding_head_shared_bytes'],'head_input_bytes':info['head_input_bytes'],'head_output_bytes':info['head_output_bytes'],
            'compressed_projection_weight_bytes':info.get('compressed_projection_weight_bytes',0),'compressed_projection_scale_bytes':info.get('compressed_projection_scale_bytes',0),'compressed_projection_pool_entries':info.get('compressed_projection_pool_entries',0),
            'proposal_io_bytes':proposal,'projection_weight_pool_bytes':info.get('projection_weight_pool_bytes',0),'projection_weight_pool_entries':info.get('projection_weight_pool_entries',0),
            'trt_runtime_allocator_live_bytes':runtime,'owned_decoder_bytes':states+requests+outputs+proposal+shared+runtime if tracked else None,
            'joined_projection_group_count':info.get('joined_projection_group_count',0),'joined_projection_group_bytes':info.get('joined_projection_group_bytes',0),
            'packed_profiles':{'prefill':self.prefill_chunk_size,'decode':self.decode_chunk_size,'single_canonical_state':True,'embedded_projection_weights':self.embedded_projection_weights,'external_compressed_weights':self.external_compressed_weights,'joined_mlp_projections':self.joined_mlp_projections,'register_delta_128':self.register_delta_128,'flat_concat':self.flat_concat,'deferred_state_commit':self.deferred_state_commit,'shared_output_storage':self.shared_output_storage,'join_decode_input_projections':self.join_decode_input_projections},
            'scope':'Retained canonical state, request/output buffers, one readonly projection pool/table and tracked TRT runtime. Excludes builder/driver/library/transient allocations; not process peak.'}

    def close(self):
        if self.state_transaction is not None:raise ValueError('Close or roll back the verifier before closing its borrowed decoder')
        try:
            if self.head is not None:self.head.close()
            for program in self.programs.values():program.close()
        finally:
            self.head=None;self.programs={};self.requests={};self.state=None;self.last_hidden=None;self.last_packed=None
            self.output_pool=None
            self.io.close();self.prefill_mode=False
