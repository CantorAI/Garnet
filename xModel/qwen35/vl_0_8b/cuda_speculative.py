# SPDX-License-Identifier: Apache-2.0
"""Experimental verified MTP session, default-off in the serving frontend.

Every emitted draft token is checked by the target. Partial target commits use
the verifier's restore/recompute correctness path. All caches reset per request;
the caller's tied embedding/head is shared, and the parent must outlive us.
"""
import json
import garnet
import tensor
from .decoder_layer import T
from .cuda_mtp import CUDAMTPProgram
from .cuda_verify import CUDATargetVerifier

def resolve_mtp_chunk(target_chunk,depth,draft_chunk=None):
    size=target_chunk if draft_chunk is None else draft_chunk
    if type(size) is not int or size not in (4,8,16) or size>target_chunk or depth+1>size:
        raise ValueError('MTP draft chunk must fit correction width and target profile')
    return size

class CUDASpeculativeSession:
    def __init__(self,decoder,checkpoint,stops,max_draft_tokens=3,head_candidates=0,prefix_replay=False,draft_chunk_size=None,draft_builder_workspace_bytes=0):
        if max_draft_tokens not in (1,3,7) or max_draft_tokens+1>decoder.chunk_size:
            raise ValueError('MTP draft depth1/3/7 must fit the target chunk')
        vocab=json.loads(decoder.io.memory_json())['vocab_size']
        if head_candidates<0 or head_candidates>min(65536,vocab):raise ValueError('Invalid MTP head candidate count')
        if type(draft_builder_workspace_bytes) is not int or draft_builder_workspace_bytes not in (0,128<<20):
            raise ValueError('MTP draft builder workspace must preserve default0 or opt in to128MiB')
        self.draft_builder_workspace_bytes=draft_builder_workspace_bytes
        self.head_candidates=head_candidates;self.candidates=None
        self.prefix_replay=prefix_replay
        self.decoder=decoder;self.io=decoder.io;self.stops=set(stops);self.depth=max_draft_tokens
        self.batch=decoder.batch;self.size=resolve_mtp_chunk(decoder.chunk_size,max_draft_tokens,draft_chunk_size);self.capacity=decoder.capacity
        self.hidden_size=decoder.config['hidden_size'];self.program=None;self.verifier=None;self.prefill_norm=None;self.buffers=[]
        self.ready=False;self.finished=False;self.poisoned=False
        self.reset_required=True
        b=self.batch;t=self.size;h=self.hidden_size
        try:
            if prefix_replay:
                if getattr(decoder,'packed_state_program',False):
                    from .cuda_packed_replay import CUDAPackedDeltaReplayVerifier
                    self.verifier=CUDAPackedDeltaReplayVerifier(decoder,checkpoint,max_draft_tokens+1)
                else:
                    from .cuda_delta_replay import CUDADeltaReplayVerifier
                    self.verifier=CUDADeltaReplayVerifier(decoder,checkpoint,max_draft_tokens+1)
            else:self.verifier=CUDATargetVerifier(decoder,checkpoint,max_draft_tokens+1)
            self.program=CUDAMTPProgram(checkpoint,decoder.config,b,self.capacity,t,decoder.projection_workspace_bytes,builder_workspace_bytes=draft_builder_workspace_bytes)
            self.program.graph.enable_cuda_graph(True)
            if getattr(decoder,'serial_graph_reuse',False):self.program.graph.enable_serial_graph_reuse(True)
            self.prefill_width=getattr(decoder,'prefill_chunk_size',decoder.chunk_size)
            if self.prefill_width!=decoder.chunk_size:
                config=decoder.config
                @T.fusion(name='qwen35_mtp_wide_target_norm',boundary='required')
                def wide_norm(value):
                    return value*T.unary_op('qwen35_rms_norm_packed',weight_name='model.language_model.norm.weight',eps=config['rms_norm_eps'])
                symbol=tensor.input('wide_mtp_target_hidden',shape=[b,self.prefill_width,h],dtype=tensor.float32)
                self.prefill_norm=garnet.compiled_graph();self.prefill_norm.build(wide_norm(symbol),[symbol],checkpoint);self.prefill_norm.enable_cuda_graph(True)
            cache=[b,decoder.config['num_key_value_heads'],self.capacity,decoder.config['head_dim']]
            self.cache_width=decoder.config['num_key_value_heads']*self.capacity*decoder.config['head_dim']
            self.target=self.io.zeros([b,t,h]);self.embedding=self.io.zeros([b,t,h]);self.key=self.io.zeros(cache);self.value=self.io.zeros(cache)
            self.position=self.io.zeros([b,t,3]);self.slot_tensor=self.io.zeros([b,t]);self.active_tensor=self.io.zeros([b,t])
            self.hidden=self.io.zeros([b,t,h]);self.last_hidden=self.io.zeros([b,h])
            self.buffers=[self.target,self.embedding,self.key,self.value,self.position,self.slot_tensor,self.active_tensor,self.hidden,self.last_hidden]
            self.slots=[0]*b;self.first_draft=None
        except:self.close();raise

    def reset(self):
        if self.program is None:raise ValueError('Speculative session is closed')
        if self.verifier.pending:self.verifier.rollback()
        self.verifier.head_evaluations=0;self.verifier.head_reuses=0;self.verifier._reuse_accepted_head=False
        self.decoder.reset()
        for value in self.buffers:self.io.zero(value)
        self.slots=[0]*self.batch;self.first_draft=None;self.candidates=None;self.ready=False;self.finished=False;self.poisoned=False
        self.reset_required=False
        self.metrics={'mtp_prefill_calls':0,'mtp_draft_calls':0,'mtp_correction_calls':0,'mtp_head_calls':0,
            'target_verification_calls':0,'target_prefix_recomputes':0,'target_prefix_replays':0,'proposed_draft_tokens':0,'accepted_draft_tokens':0,'iterations':[]}

    def _call(self,flags,positions,predict,stage):
        count=len(flags);active=[];slots=[];pos=[];next_slots=list(self.slots)
        for b in range(self.batch):
            for t in range(self.size):
                flag=flags[t][b] if t<count else 0
                if flag and next_slots[b]>=self.capacity:raise ValueError('MTP cache capacity exhausted')
                active.append(flag);slots.append(next_slots[b]);next_slots[b]+=flag
                pos.extend(positions[b][t] if t<count else [0,0,0])
        if getattr(self.decoder,'resident_controls',False):self.io.update_request(self.active_tensor,self.position,self.slot_tensor,active,pos,slots)
        else:
            self.io.update(self.active_tensor,active);self.io.update(self.slot_tensor,slots);self.io.update(self.position,pos)
        packed=self.program.run([self.target,self.embedding,self.key,self.value,self.position,self.slot_tensor,self.active_tensor])
        self.io.slice_into(packed,0,self.hidden);self.io.slice_into(packed,self.size*self.hidden_size,self.key)
        self.io.slice_into(packed,self.size*self.hidden_size+self.cache_width,self.value)
        self.io.copy_last_chunk(self.hidden,self.active_tensor,self.last_hidden);self.slots=next_slots
        self.metrics['mtp_'+stage+'_calls']+=1
        if predict:
            self.metrics['mtp_head_calls']+=1
            return list(self.io.predict_indexed(self.last_hidden,self.candidates) if self.head_candidates else self.io.predict(self.last_hidden))
        return None

    def _step(self,normalized,tokens,active,positions,predict,stage):
        self.io.zero(self.target);self.io.write_chunk_token(normalized,self.target,0)
        if getattr(self.decoder,'resident_controls',False):self.io.embedding_chunk([tokens],self.embedding)
        else:
            self.io.zero(self.embedding);self.io.write_chunk_token(self.io.embedding(tokens),self.embedding,0)
        return self._call([active],[[list(p)] for p in positions],predict,stage)

    def _prefill_hook(self,hidden_tokens,active_tokens,positions,predict=True):
        prediction=self._original_chunk(hidden_tokens,active_tokens,positions,predict)
        if predict and self._prefill_emit is not None:self._prefill_emit(list(prediction))
        count=len(hidden_tokens);start=self._prefill_index
        graph=self.prefill_norm if self.prefill_norm is not None else self.verifier.graph
        normalized=graph.run_device([self.decoder.hidden])
        # A wide target prefill feeds the one existing small MTP engine in
        # slices. Its weights/cache remain single-copy; no second draft engine.
        for local in range(0,count,self.size):
            self.io.slice_into(normalized,local*self.hidden_size,self.target)
            self.io.zero(self.embedding);flags=[];shifted_positions=[[] for _ in range(self.batch)]
            for t in range(min(self.size,count-local)):
                global_index=start+local+t;next_index=global_index+1
                if next_index<len(self._prefill_hidden):self.io.write_chunk_token(self._prefill_hidden[next_index],self.embedding,t)
                flags.append([int(global_index<length-1) for length in self._prefill_lengths])
                for b in range(self.batch):
                    shifted_positions[b].append(self._prefill_positions[b][next_index] if next_index<len(self._prefill_positions[b]) else [0,0,0])
            self._call(flags,shifted_positions,False,'prefill')
        self._prefill_index+=count
        return prediction

    def prefill(self,hidden_tokens,positions,lengths,next_positions,on_first_token=None):
        if self.program is None or self.reset_required or self.poisoned or self.ready:raise ValueError('Reset an open speculative session before prefill')
        self._prefill_hidden=hidden_tokens;self._prefill_positions=positions;self._prefill_lengths=list(lengths);self._prefill_index=0
        self._prefill_emit=on_first_token
        self._had_chunk_override='chunk_hidden' in self.decoder.__dict__
        self._chunk_override=self.decoder.__dict__.get('chunk_hidden')
        self._original_chunk=self.decoder.chunk_hidden;self.decoder.chunk_hidden=self._prefill_hook
        try:
            self.current=list(self.decoder.prefill_sequence(hidden_tokens,positions,lengths))
        except:self.poisoned=True;raise
        finally:
            if self._had_chunk_override:self.decoder.chunk_hidden=self._chunk_override
            else:delattr(self.decoder,'chunk_hidden')
            self._original_chunk=None;self._chunk_override=None
            self._prefill_hidden=None;self._prefill_positions=None
            self._prefill_emit=None
        try:
            if self.head_candidates:self.candidates=self.io.head_candidates(list(range(self.batch)),self.head_candidates,False)
            normalized=self.decoder.head.run_device([self.decoder.last_hidden])
            self.first_draft=self._step(normalized,self.current,[1]*self.batch,[[p]*3 for p in next_positions],True,'prefill')
            self.current_positions=list(next_positions);self.ready=True
            assert self.slots==self.decoder.slots,'Shifted MTP prompt cache length differs'
            return list(self.current)
        except:self.poisoned=True;raise

    def _draft(self,current,active,remaining,positions):
        count=min(self.depth,max(remaining[b]-1 for b in range(self.batch) if active[b]))
        proposals=[list(current)];next_draft=list(self.first_draft);stopped=[False]*self.batch
        for j in range(1,count+1):
            proposals.append([next_draft[b] if active[b] and not stopped[b] else 0 for b in range(self.batch)])
            advance=[]
            for b in range(self.batch):
                eligible=active[b] and not stopped[b] and j<remaining[b]
                if eligible:self.metrics['proposed_draft_tokens']+=1
                if eligible and next_draft[b] in self.stops:stopped[b]=True
                advance.append(int(active[b] and not stopped[b] and j<remaining[b]-1))
            if j<count and any(advance):
                next_draft=self._step(self.last_hidden,proposals[-1],advance,[[p+j]*3 for p in positions],True,'draft')
        return proposals

    def _correct(self,base_slots,positions,counts,proposals,next_tokens,predict):
        self.slots=list(base_slots);self.io.slice_into(self.verifier.normalized_packed,0,self.target)
        resident=getattr(self.decoder,'resident_controls',False)
        if not resident:self.io.zero(self.embedding)
        n=len(proposals);flags=[];shifted_positions=[[] for _ in range(self.batch)];token_rows=[]
        for t in range(n):
            ids=[]
            for b in range(self.batch):
                if t<counts[b]-1:ids.append(proposals[t+1][b])
                elif t==counts[b]-1:ids.append(next_tokens[b])
                else:ids.append(0)
                shifted_positions[b].append([positions[b]+t+1]*3)
            if resident:token_rows.append(ids)
            else:self.io.write_chunk_token(self.io.embedding(ids),self.embedding,t)
            flags.append([int(t<count) for count in counts])
        if resident:self.io.embedding_chunk(token_rows,self.embedding)
        return self._call(flags,shifted_positions,predict,'correction')

    def generate(self,limit,on_token=None):
        if not self.ready or self.finished or self.poisoned:raise ValueError('Prefill a fresh speculative request first')
        if limit<1 or any(slot+limit-1>self.capacity for slot in self.decoder.slots):raise ValueError('Speculative output exceeds target capacity')
        b=self.batch;active=[1]*b;outputs=[[] for _ in range(b)];sampled=[0]*b;reasons=['length']*b;stop_ids=[None]*b
        try:
            while any(active):
                for row in range(b):
                    if not active[row]:continue
                    if self.current[row] in self.stops:
                        sampled[row]+=1;active[row]=0;reasons[row]='eos';stop_ids[row]=self.current[row]
                        if on_token is not None:on_token(row,self.current[row],sampled[row])
                    elif sampled[row]+1==limit:
                        outputs[row].append(self.current[row]);sampled[row]+=1;active[row]=0
                        if on_token is not None:on_token(row,self.current[row],sampled[row])
                if not any(active):break
                working=list(active);remaining=[limit-n for n in sampled]
                base_slots=list(self.decoder.slots);base_positions=list(self.current_positions)
                proposals=self._draft(self.current,working,remaining,base_positions);n=len(proposals)
                flags=[];stopped=[False]*b
                for t in range(n):
                    row_flags=[]
                    for row in range(b):
                        if proposals[t][row] in self.stops:stopped[row]=True
                        row_flags.append(int(working[row] and not stopped[row] and t<remaining[row]-1))
                    flags.append(row_flags)
                positions=[]
                for p in base_positions:positions.append([[p+t]*3 for t in range(n)])
                verified=self.verifier.verify(proposals,flags,positions);self.metrics['target_verification_calls']+=1
                counts=[0]*b
                for row in range(b):
                    if not working[row]:continue
                    outputs[row].append(self.current[row]);sampled[row]+=1
                    if on_token is not None:on_token(row,self.current[row],sampled[row])
                    for j in range(1,n):
                        proposed=proposals[j][row];actual=verified[row][j-1]
                        if proposed!=actual:
                            counts[row]=j;self.current[row]=actual;break
                        self.metrics['accepted_draft_tokens']+=1;sampled[row]+=1;self.current[row]=proposed
                        if on_token is not None:on_token(row,proposed,sampled[row])
                        if proposed in self.stops:
                            active[row]=0;reasons[row]='eos';stop_ids[row]=proposed;counts[row]=j;break
                        outputs[row].append(proposed)
                        if sampled[row]==limit:active[row]=0;counts[row]=j;break
                    else:
                        counts[row]=n;self.current[row]=verified[row][n-1]
                if counts!=self.verifier.submitted:self.metrics['target_prefix_replays' if self.prefix_replay else 'target_prefix_recomputes']+=1
                committed=self.verifier.commit_prefix(counts)
                if self.head_candidates:
                    selected=[row*self.verifier.max_tokens+max(counts[row],1)-1 for row in range(b)]
                    self.candidates=self.io.head_candidates(selected,self.head_candidates,True)
                for row in range(b):
                    if working[row] and active[row]:self.current[row]=committed[row]
                    self.current_positions[row]+=counts[row]
                self.first_draft=self._correct(base_slots,base_positions,counts,proposals,self.current,any(active))
                assert self.slots==self.decoder.slots,'Corrected MTP cache length differs'
                self.metrics['iterations'].append({'processed_target_counts':counts,'sampled_so_far':list(sampled)})
            self.finished=True
            self.metrics['target_forward_calls']=self.metrics['target_verification_calls']+self.metrics['target_prefix_recomputes']
            self.metrics['target_head_evaluations']=self.verifier.head_evaluations
            self.metrics['target_head_reuses']=self.verifier.head_reuses
            return {'generated_token_ids':outputs,'sampled_tokens_including_stop':sampled,'finish_reasons':reasons,
                'stop_token_ids':stop_ids,'decode_steps':max(sampled)-1,'speculative_metrics':self.metrics}
        except:
            self.poisoned=True
            if self.verifier.pending:self.verifier.rollback()
            raise

    def memory_info(self):
        if self.program is None:raise ValueError('Speculative session is closed')
        graph=json.loads(self.program.graph.stats_json());helper=self.verifier.memory_info()
        buffers=sum(self.io.tensor_bytes(value) for value in self.buffers)
        live=graph['trt_allocator_live_bytes'] if graph['allocator_tracking_enabled'] else None
        prefill=json.loads(self.prefill_norm.stats_json()) if self.prefill_norm is not None else None
        prefill_owned=prefill['device_output_bytes']+prefill['trt_allocator_live_bytes'] if prefill is not None and prefill['allocator_tracking_enabled'] else 0 if prefill is None else None
        owned=buffers+graph['device_output_bytes']+live+helper['owned_helper_bytes']+prefill_owned if live is not None and helper['owned_helper_bytes'] is not None and prefill_owned is not None else None
        return {'mtp_buffers_bytes':buffers,'mtp_graph':graph,'target_verifier':helper,'owned_speculative_bytes':owned,'draft_builder_workspace_bytes':self.draft_builder_workspace_bytes,
            'wide_target_norm_graph':prefill,'target_prefill_chunk':self.prefill_width,'draft_chunk':self.size,
            'scope':'Session-only buffers/engines/backups. Caller owns the one shared table/head buffers; not process peak'}

    def close(self):
        try:
            if self.verifier is not None:self.verifier.close()
        finally:
            try:
                if self.prefill_norm is not None:self.prefill_norm.close()
            finally:
                if self.program is not None:self.program.close()
                self.program=None;self.verifier=None;self.prefill_norm=None;self.buffers=[];self.ready=False
                self.target=self.embedding=self.key=self.value=None
                self.candidates=None
                self.position=self.slot_tensor=self.active_tensor=self.hidden=self.last_hidden=None
                self.decoder=None;self.io=None
