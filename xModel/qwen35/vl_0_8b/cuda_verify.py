# SPDX-License-Identifier: Apache-2.0
"""Experimental target verification over a borrowed decoder, not serving.

The first implementation restores/recomputes partial accepted prefixes. This
establishes a correctness path before adding faster prefix-state selection.
Close this helper before closing its parent decoder. Returned normalized packed
storage is reused by the next verification or partial-prefix recomputation.
"""
import json
import garnet
import tensor
from .decoder_layer import T

class CUDATargetVerifier:
    def __init__(self,decoder,checkpoint,max_tokens=4):
        if getattr(decoder,'deferred_state_commit',False) and not getattr(self,'supports_deferred_state_commit',False):
            raise ValueError('Deferred canonical state requires the packed replay verifier')
        if max_tokens<2 or max_tokens>decoder.chunk_size:raise ValueError('Verification requires2..chunk_size tokens')
        if decoder.poisoned or decoder.head is None:raise ValueError('Verification requires a ready decoder')
        self.decoder=decoder;self.io=decoder.io;self.max_tokens=max_tokens;self.pending=False;self.graph=None
        self.backups=[];self.head_input=None;self.normalized_packed=None
        self._reuse_accepted_head=False;self.head_evaluations=0;self.head_reuses=0
        if hasattr(decoder,'verification_tensors'):self.tensors=decoder.verification_tensors()
        else:
            self.tensors=[decoder.hidden,decoder.last_hidden,decoder.active,decoder.position,decoder.slot_tensor]
            self.tensors += [layer[key] for layer in decoder.layers for key in ('state1','state2')]
        b=decoder.batch;size=decoder.chunk_size;h=decoder.config['hidden_size']
        self.graph=garnet.compiled_graph()
        @T.fusion(name='qwen35_verification_norm',boundary='required')
        def normalize(value):
            return value*T.unary_op('qwen35_rms_norm_packed',weight_name='model.language_model.norm.weight',eps=decoder.config['rms_norm_eps'])
        try:
            symbol=tensor.input('verification_hidden',shape=[b,size,h],dtype=tensor.float32)
            self.graph.build(normalize(symbol),[symbol],checkpoint);self.graph.enable_cuda_graph(True)
            self.graph.enable_serial_graph_reuse(bool(getattr(decoder,'serial_target_graph_reuse',False)))
            self.backups=[self.io.clone(value) for value in self.tensors]
            self.head_input=self.io.zeros([b,max_tokens,h])
        except:
            self.graph.close();self.graph=None;self.backups=[];raise

    def _ready(self):
        if self.graph is None or self.decoder.head is None:raise ValueError('Verification helper or parent is closed')
        if self.decoder.poisoned:raise ValueError('Reset a poisoned target before verification')

    def _begin(self):
        self._ready()
        if self.pending:raise ValueError('Commit or roll back the pending verification first')
        for source,backup in zip(self.tensors,self.backups):self.io.copy_into(source,backup)
        self.saved_slots=list(self.decoder.slots);self.pending=True

    def _read(self):
        self.normalized_packed=self.graph.run_device([self.decoder.hidden])
        self.io.slice_into(self.normalized_packed,0,self.head_input)
        if self._reuse_accepted_head:
            if not isinstance(getattr(self,'predictions',None),list) or len(self.predictions)!=self.decoder.batch*self.max_tokens:
                raise ValueError('Accepted target head cache is unavailable')
            self.head_reuses+=1
        else:
            self.predictions=list(self.io.predict_tokens(self.head_input));self.head_evaluations+=1
        return [self.predictions[b*self.max_tokens:(b+1)*self.max_tokens] for b in range(self.decoder.batch)]

    def verify(self,tokens,active,positions):
        self._ready();n=len(tokens);b=self.decoder.batch
        if not 1<=n<=self.max_tokens or len(active)!=n or any(len(row)!=b for row in tokens+active):
            raise ValueError('Invalid verification token/control shape')
        if any(flag not in (0,1) for row in active for flag in row):raise ValueError('Invalid verification active flag')
        counts=[]
        for row in range(b):
            flags=[active[t][row] for t in range(n)];count=sum(flags)
            if flags!=[1]*count+[0]*(n-count):raise ValueError('Verification active rows must be contiguous prefixes')
            counts.append(count)
        if len(positions)!=b:raise ValueError('Verification positions require batch/time/THW')
        for row in positions:
            if len(row)!=n:raise ValueError('Verification positions require batch/time/THW')
            for position in row:
                if len(position)!=3:raise ValueError('Verification positions require batch/time/THW')
        self._begin();self.submitted=counts;self.count=n
        self.positions=[[list(p) for p in row] for row in positions]
        try:
            if getattr(self.decoder,'resident_controls',False):
                self.token_ids=[list(row) for row in tokens]
                self.decoder.chunk_tokens(self.token_ids,[list(row) for row in active],self.positions,False)
            else:
                self.embeddings=[self.io.embedding(list(row)) for row in tokens]
                self.decoder.chunk_hidden(self.embeddings,[list(row) for row in active],self.positions,False)
            return self._read()
        except:
            self.rollback();raise

    def rollback(self):
        if not self.pending:raise ValueError('No pending verification to roll back')
        for backup,destination in zip(self.backups,self.tensors):self.io.copy_into(backup,destination)
        self.decoder.slots=list(self.saved_slots);self.decoder.poisoned=False;self.pending=False

    def commit_prefix(self,counts):
        if not self.pending:raise ValueError('No pending verification to commit')
        if len(counts)!=self.decoder.batch or any(not isinstance(n,int) or n<0 or n>limit for n,limit in zip(counts,self.submitted)):
            raise ValueError('Accepted prefix exceeds submitted tokens')
        if counts!=self.submitted:
            self.rollback();self.pending=True
            flags=[]
            for t in range(self.count):flags.append([int(t<counts[b]) for b in range(self.decoder.batch)])
            try:
                if getattr(self.decoder,'resident_controls',False):self.decoder.chunk_tokens(self.token_ids,flags,self.positions,False)
                else:self.decoder.chunk_hidden(self.embeddings,flags,self.positions,False)
                self._read()
            except:
                self.rollback();raise
        self.pending=False
        return [self.predictions[b*self.max_tokens+n-1] if n else None for b,n in enumerate(counts)]

    def memory_info(self):
        self._ready();stats=json.loads(self.graph.stats_json())
        buffers=sum(self.io.tensor_bytes(value) for value in self.backups)+self.io.tensor_bytes(self.head_input)
        live=stats['trt_allocator_live_bytes'] if stats['allocator_tracking_enabled'] else None
        return {'backup_and_head_input_bytes':buffers,'norm_graph':stats,
            'target_head_evaluations':self.head_evaluations,'target_head_reuses':self.head_reuses,
            'owned_helper_bytes':buffers+stats['device_output_bytes']+live if live is not None else None,
            'scope':'Helper-owned backups/normalization only; shared table/head buffers counted by parent IO; not process peak'}

    def close(self):
        if self.graph is not None:
            try:
                if self.pending:self.rollback()
            finally:
                self.graph.close();self.graph=None;self.backups=[];self.head_input=None;self.normalized_packed=None;self.tensors=[]
                self.decoder=None;self.io=None
