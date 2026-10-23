# SPDX-License-Identifier: Apache-2.0
"""Single-graph prefix state commit over retained authoritative projections."""
import json
import garnet
import tensor
from .decoder_layer import T
from .cuda_verify import CUDATargetVerifier
from .cuda_packed_program import packed_view,concat,flat_concat

class CUDAPackedDeltaReplayVerifier(CUDATargetVerifier):
    supports_deferred_state_commit=True
    def __init__(self,decoder,checkpoint,max_tokens=4,native_prefix_commit=None):
        if not getattr(decoder,'packed_state_program',False) or not decoder.retain_linear_inputs:
            raise ValueError('Packed replay requires retained packed decoder projections')
        self.replay_graph=None;self.cache_spans=None;self.native_prefix=None
        self.deferred=bool(getattr(decoder,'deferred_state_commit',False))
        if native_prefix_commit is None:native_prefix_commit=bool(getattr(decoder,'native_prefix_commit',False))
        self.reuse_verified_head=bool(getattr(decoder,'reuse_verified_head',False))
        if self.reuse_verified_head and not native_prefix_commit:
            raise ValueError('Verified head reuse requires the native packed-prefix profile')
        if native_prefix_commit and (not self.deferred or not decoder.register_delta_128):
            raise ValueError('Native prefix commit requires deferred state and register128 arithmetic')
        super().__init__(decoder,checkpoint,max_tokens)
        try:
            p=decoder.programs[decoder.decode_chunk_size];b=decoder.batch;t=decoder.decode_chunk_size
            if native_prefix_commit:
                self.native_prefix=garnet.cuda_packed_prefix()
                self.native_prefix.build(checkpoint,json.dumps({'config':decoder.config,'layout':p.layout,
                    'batch':b,'tokens':t,'capacity':decoder.capacity,'state_width':p.state_width,
                    'packed_width':p.output_width,'hidden_width':p.hidden_width}))
                return
            self.cache_spans=self.io.zeros([decoder.batch,2])
            symbols=[tensor.input('whole_replay_'+str(i),shape=s,dtype=tensor.float32)
                for i,s in enumerate(([b,p.output_width],[b,p.state_width],[b,t],[b,2]))]
            self.replay_graph=garnet.compiled_graph()
            self.replay_graph.build(self._forward(p)(*symbols),symbols,checkpoint);self.replay_graph.enable_cuda_graph(True)
            self.replay_graph.enable_serial_graph_reuse(bool(getattr(decoder,'serial_target_graph_reuse',False)))
        except:self.close();raise

    def _forward(self,program):
        c=self.decoder.config;b=self.decoder.batch;t=self.decoder.decode_chunk_size;h=c['hidden_size'];join=flat_concat if program.flat_concat else concat
        @T.fusion(name='qwen35_whole_prefix_replay',boundary='required')
        def replay(packed,backup,active,spans):
            states=[]
            for layer in program.layout:
                offset=layer['offset'];w1=layer['state1_width'];w2=layer['state2_width']
                old1=packed_view(backup,offset,w1,layer['state1_shape']);old2=packed_view(backup,offset+w1,w2,layer['state2_shape'])
                if layer['kind']=='linear_attention':
                    # Existing replay lowering consumes the same retained payload
                    # layout as a per-layer output. Unused hidden/state fields
                    # are compiler-pruned; no target projection is rerun.
                    local=join([packed_view(packed,0,t*h,[b,t*h]),packed_view(backup,offset,w1+w2,[b,w1+w2]),
                        packed_view(packed,layer['payload_offset'],layer['payload_width'],[b,layer['payload_width']])])
                    bound=local*T.binary_op('qwen35_conv_bind_state')*old1
                    bound=bound*T.binary_op('qwen35_delta_bind_state')*old2
                    bound=bound*T.binary_op('qwen35_conv_bind_active')*active
                    states.append(bound*T.unary_op('qwen35_linear_replay_chunk',layer_prefix='model.language_model.layers.'+str(layer['index']),
                        hidden_dim=h,tokens=t,key_heads=c['linear_num_key_heads'],value_heads=c['linear_num_value_heads'],
                        key_dim=c['linear_key_head_dim'],value_dim=c['linear_value_head_dim'],eps=c['rms_norm_eps'],register_delta_128=int(program.register_delta_128)))
                else:
                    for start,width,shape,old in ((offset,w1,layer['state1_shape'],old1),(offset+w1,w2,layer['state2_shape'],old2)):
                        current=packed_view(packed,program.hidden_width+start,width,shape)
                        bound=current*T.binary_op('qwen35_cache_bind_backup')*old
                        bound=bound*T.binary_op('qwen35_cache_bind_spans')*spans
                        restored=bound*T.unary_op('qwen35_restore_kv_prefix')
                        states.append(restored*T.unary_op('reshape',shape=[b,width]))
            return join(states)
        return replay

    def _begin(self):
        if not self.deferred:return super()._begin()
        self._ready()
        if self.pending:raise ValueError('Commit or roll back pending verification first')
        self.decoder._claim_state_transaction(self)
        try:super()._begin()
        except:
            self.decoder._release_state_transaction(self)
            raise

    def rollback(self):
        if not self.deferred:return super().rollback()
        if not self.pending:raise ValueError('No pending verification to roll back')
        try:super().rollback()
        except:
            self.decoder.poisoned=True;self.pending=False
            raise
        finally:self.decoder._release_state_transaction(self)

    def _write_canonical(self,source,packed):
        if packed:self.io.slice_into(source,self.decoder.programs[self.decoder.decode_chunk_size].hidden_width,self.decoder.state)
        else:self.io.copy_into(source,self.decoder.state)

    def _commit_canonical(self,source,packed):
        # This is the final write, after all recoverable validation/head work.
        # An error during that write has unknown completion; poison instead of
        # pretending an old canonical backup still exists.
        try:self._write_canonical(source,packed)
        except:
            self.decoder.poisoned=True;self.pending=False
            self.decoder._release_state_transaction(self)
            raise

    def _write_native(self,spans):
        self.native_prefix.commit(self.decoder.last_packed,self.decoder.state,spans)

    def _commit_native(self,spans):
        try:self._write_native(spans)
        except:
            self.decoder.poisoned=True;self.pending=False
            self.decoder._release_state_transaction(self)
            raise

    def commit_prefix(self,counts):
        if not self.pending:raise ValueError('No pending verification to commit')
        d=self.decoder;b=d.batch
        if len(counts)!=b or any(not isinstance(n,int) or n<0 or n>limit for n,limit in zip(counts,self.submitted)):
            raise ValueError('Accepted prefix exceeds submitted tokens')
        if counts==self.submitted:
            predictions=[self.predictions[row*self.max_tokens+n-1] if n else None for row,n in enumerate(counts)]
            if self.deferred:self._commit_canonical(d.last_packed,True)
            self.pending=False
            if self.deferred:d._release_state_transaction(self)
            return predictions
        flags=[];slots=[];spans=[]
        for row in range(b):
            slot=self.saved_slots[row];spans.extend([slot,counts[row]])
            for t in range(d.chunk_size):
                active=int(t<counts[row]);flags.append(active);slots.append(slot);slot+=active
        try:
            self.io.update(d.active,flags);self.io.update(d.slot_tensor,slots)
            if self.native_prefix is None:
                self.io.update(self.cache_spans,spans)
                old_state=d.state if self.deferred else self.backups[5]
                result=self.replay_graph.run_device([d.last_packed,old_state,d.active,self.cache_spans])
                if not self.deferred:self.io.copy_into(result,d.state)
            self.io.mask_chunk(d.hidden,d.active)
            self.io.copy_into(self.backups[1],d.last_hidden);self.io.copy_last_chunk(d.hidden,d.active,d.last_hidden)
            next_slots=[self.saved_slots[row]+counts[row] for row in range(b)]
            if not self.deferred:d.slots=next_slots;d.poisoned=False
            # Accepted hidden rows already have authoritative target logits.
            # Refresh masked normalization for MTP correction, while preserving
            # those accepted logits. Explicit zero-prefix commits recompute.
            self._reuse_accepted_head=self.reuse_verified_head and all(n>0 or submitted==0 for n,submitted in zip(counts,self.submitted))
            try:self._read()
            finally:self._reuse_accepted_head=False
            predictions=[self.predictions[row*self.max_tokens+n-1] if n else None for row,n in enumerate(counts)]
            if self.deferred:
                if self.native_prefix is not None:self._commit_native(spans)
                else:self._commit_canonical(result,False)
                d.slots=next_slots;d.poisoned=False
            self.pending=False
            if self.deferred:d._release_state_transaction(self)
            return predictions
        except:
            if self.pending:self.rollback()
            raise

    def memory_info(self):
        result=super().memory_info()
        stats=json.loads(self.replay_graph.stats_json()) if self.replay_graph is not None else None
        native=json.loads(self.native_prefix.stats_json()) if self.native_prefix is not None else None
        spans=self.io.tensor_bytes(self.cache_spans) if self.cache_spans is not None else 0
        extra=native['owned_bytes'] if native is not None else (stats['device_output_bytes']+stats['trt_allocator_live_bytes'] if stats['allocator_tracking_enabled'] else None)
        result.update({'packed_replay_graph':stats,'native_prefix_commit':native,'cache_span_bytes':spans,
            'reuse_verified_head':self.reuse_verified_head,
            'head_reuse_scope':'Accepted rows within one verified window; masked normalization is refreshed. Active zero-prefix rows force a full read.',
            'partial_commit_method':'native_inplace_state_only_replay' if native is not None else 'single_graph_delta_replay_and_kv_selection'})
        result.update({'deferred_state_commit':self.deferred,'canonical_state_backup_bytes':0 if self.deferred else self.io.tensor_bytes(self.decoder.state),
            'canonical_state_scope':'Borrowed and unchanged until final commit; parent mutation is blocked while pending. Final-write failures poison the decoder.' if self.deferred else 'Owned full canonical backup'})
        result['owned_helper_bytes']=result['owned_helper_bytes']+spans+extra if extra is not None and result['owned_helper_bytes'] is not None else None
        return result

    def close(self):
        try:super().close()
        finally:
            if self.replay_graph is not None:self.replay_graph.close()
            if self.native_prefix is not None:self.native_prefix.close()
            self.replay_graph=None;self.native_prefix=None;self.cache_spans=None
