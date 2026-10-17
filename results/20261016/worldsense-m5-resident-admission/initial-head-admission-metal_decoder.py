# SPDX-License-Identifier: Apache-2.0
"""Garnet Metal decoder: reuse explicit-state serving, project the head on GPU."""
import garnet,tensor,time
from .cpu_decoder import CPUDecoder
from .decoder_layer import T,norm,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer

class MetalDecoder(CPUDecoder):
    def __init__(self,checkpoint,config,batch=1,capacity=256,bf16_head=True,resident=True,bf16_projections=True):
        self.session=None;self.bf16_projections=bf16_projections;self.resident_construction_requested=resident
        backend=garnet.backend_info();self.joined_prefill_enabled='joined_mps_prefill' in backend
        self.packed_prefill_norm_enabled='packed_prefill_norm' in backend
        self.global_half_alpha_enabled='global_half_alpha' in backend
        self.joined_global_mlp_enabled='joined_global_mlp' in backend
        if not backend.startswith('Garnet Metal/MPS '):
            raise ValueError('MetalDecoder requires the native Garnet Metal package')
        CPUDecoder.__init__(self,checkpoint,config,batch,capacity)
        try:
            self.head.close()
            symbol=tensor.input('metal_final_hidden',shape=[batch,config['hidden_size']],dtype=tensor.float32)
            @T.fusion(name='qwen35_metal_tied_head',boundary='required')
            def head(hidden):
                return norm(hidden,'model.language_model.norm',config) * T.unary_op('qwen35_tied_head',bf16_weight=1 if bf16_head else 0)
            self.head=garnet.compiled_graph();self.head.build(head(symbol),[symbol],checkpoint)
            if resident:self.enable_resident()
        except:
            self.close();raise

    def _build_graph(self,graph,output,symbols,checkpoint):
        if self.bf16_projections:
            if self.resident_construction_requested:graph.build_bf16_resident(output,symbols,checkpoint)
            else:graph.build_bf16(output,symbols,checkpoint)
        else:graph.build(output,symbols,checkpoint)

    def enable_resident(self):
        if self.session is not None:return
        if any(self.slots):raise ValueError('reset before enabling GPU resident state')
        session=garnet.decoder_session()
        try:session.build([layer['graph'] for layer in self.layers],self.head)
        except:session.close();raise
        self.session=session
        # Resident state replaces the legacy host copies. Explicit snapshots are
        # available through session.snapshot(layer), without stale host values.
        for layer in self.layers:layer['state1']=None;layer['state2']=None

    def reset(self):
        if getattr(self,'session',None) is None:CPUDecoder.reset(self)
        else:
            self.session.reset();self.slots=[0 for i in range(self.batch)];self.poisoned=False

    def step_hidden(self,hidden,active=None,positions=None,predict=True):
        if self.session is None:return CPUDecoder.step_hidden(self,hidden,active,positions,predict)
        if self.poisoned:raise ValueError('reset request state after failed decoder step')
        if active is None:active=[1 for i in range(self.batch)]
        if len(active)!=self.batch or any(a not in (0,1) for a in active):raise ValueError('invalid active rows')
        if any(active[b] and self.slots[b]>=self.capacity for b in range(self.batch)):raise ValueError('context capacity exhausted')
        if positions is None:positions=[[self.slots[b]]*3 for b in range(self.batch)]
        if len(positions)!=self.batch or any(len(p)!=3 for p in positions):raise ValueError('positions must be batch x THW')
        flag=tensor.tensor(active,shape=[self.batch],dtype=tensor.float32)
        position=tensor.tensor([v for row in positions for v in row],shape=[self.batch,3],dtype=tensor.float32)
        slots=tensor.tensor(self.slots,shape=[self.batch],dtype=tensor.float32)
        self.poisoned=True
        prediction=self.session.run([layer['graph'] for layer in self.layers],hidden,position,slots,flag,0 if predict else -1)
        self.last_session_ms=self.session.last_inference_ms;self.last_layer_ms=[];self.last_head_ms=0
        for b in range(self.batch):self.slots[b]+=active[b]
        self.poisoned=False
        return prediction if predict else None

    def _prefill_profiles(self):
        if 'qwen35_extended_prefill' in garnet.backend_info():return (2,4,8,16,32,40,64,80)
        return (2,4,8,16,32,64)

    def _prefill_chunk_size(self,remaining,maximum):
        chunk=maximum
        if maximum>16 and remaining<maximum:
            chunk=2
            while chunk<remaining:chunk*=2
        return min(chunk,maximum)

    def enable_prefill(self,chunk=16,projection='shared',linear_chunk=False):
        if chunk not in self._prefill_profiles():raise ValueError('Unsupported Metal prefill chunk')
        if projection not in ('shared','mps'):raise ValueError('Prefill projection must be shared or mps')
        wide_batch=self.batch==2 and chunk==80 and 'qwen35_wide_batch_prefill' in garnet.backend_info()
        if chunk in (40,80) and (projection!='shared' or not linear_chunk or self.session is None or (self.batch*chunk>128 and not wide_batch) or self.config['hidden_size']>1024):
            raise ValueError('Extended Qwen3.5 chunks require resident shared chunk recurrence, an admitted row count, and hidden size <= 1024')
        if hasattr(self,'prefill_graphs') and (getattr(self,'chunk',None)!=chunk or getattr(self,'prefill_projection','shared')!=projection or getattr(self,'linear_chunk',False)!=linear_chunk):
            for graph in self.prefill_graphs.values():graph.close()
            self.prefill_graphs={}
        self.chunk=chunk;self.prefill_projection=projection;self.linear_chunk=linear_chunk
        if not hasattr(self,'prefill_graphs'):self.prefill_graphs={}

    def _build_prefill_graph(self,graph,output,symbols,layer_graph):
        # Experimental MPS policy expands the original BF16 values exactly to
        # FP32 once. Decode still uses its existing resident BF16 weights.
        # This consumes additional memory and is not the default policy.
        if self.prefill_projection=='mps':graph.build(output,symbols,self.checkpoint)
        elif self.session is not None:graph.build_shared_resident(output,symbols,self.checkpoint,layer_graph)
        else:graph.build_shared(output,symbols,self.checkpoint,layer_graph)

    def _prefill_layer(self,hidden,state1,state2,positions,slots,active,config,idx,batch,chunk,capacity,kind):
        from .prefill_layer import PrefillLayer
        return PrefillLayer(hidden,state1,state2,positions,slots,active,config,idx,batch,chunk,capacity,kind,parallel_attention=True,parallel_linear=self.linear_chunk)

    def prefill_sequence(self,hidden_tokens,positions,predict=True):
        if self.session is not None:return self._resident_prefill(hidden_tokens,positions,predict)
        if self.poisoned:raise ValueError('reset after failed prefill')
        count=len(hidden_tokens);chunk=self.chunk;batch=self.batch;config=self.config;capacity=self.capacity
        if count<1 or any(slot+count>capacity for slot in self.slots):raise ValueError('invalid prefill length/capacity')
        if len(positions)!=batch or any(len(row)!=count or any(len(p)!=3 for p in row) for row in positions):raise ValueError('invalid prefill THW positions')
        width=config['hidden_size'];self.poisoned=True;self.prefill_compile_ms=0;self.prefill_inference_ms=0
        for idx,layer in enumerate(self.layers):
            kind=layer['kind'];shapes=[[batch,chunk,width],layer['shapes'][1],layer['shapes'][2],
                [batch,chunk,3],[batch,chunk],[batch,chunk]]
            symbols=[tensor.input('prefill_'+str(idx)+'_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
            @T.fusion(name='qwen35_metal_prefill',boundary='required')
            def forward(hidden,state1,state2,positions,slots,active):
                return self._prefill_layer(hidden,state1,state2,positions,slots,active,config,idx,batch,chunk,capacity,kind)
            key=str(chunk)+'-'+str(idx)
            graph=self.prefill_graphs.get(key);next_hidden=[]
            if graph is None:
                graph=garnet.compiled_graph()
                try:self._build_prefill_graph(graph,forward(*symbols),symbols,layer['graph'])
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

    def _resident_prefill_graphs(self,chunk):
        batch=self.batch;config=self.config;capacity=self.capacity;width=config['hidden_size'];graphs=[]
        for idx,layer in enumerate(self.layers):
            shapes=[[batch,chunk,width],layer['shapes'][1],layer['shapes'][2],[batch,chunk,3],[batch,chunk],[batch,chunk]]
            key=str(chunk)+'-'+str(idx);graph=self.prefill_graphs.get(key)
            if graph is None:
                symbols=[tensor.input('resident_prefill_'+str(idx)+'_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
                kind=layer['kind']
                @T.fusion(name='qwen_metal_resident_prefill',boundary='required')
                def forward(hidden,state1,state2,position,slots,active):
                    return self._prefill_layer(hidden,state1,state2,position,slots,active,config,idx,batch,chunk,capacity,kind)
                graph=garnet.compiled_graph()
                try:self._build_prefill_graph(graph,forward(*symbols),symbols,layer['graph'])
                except:graph.close();raise
                self.prefill_graphs[key]=graph;self.prefill_compile_ms+=graph.compile_ms
            graphs.append(graph)
        return graphs

    def _resident_prefill(self,hidden_tokens,positions,predict):
        if self.poisoned:raise ValueError('reset after failed prefill')
        count=len(hidden_tokens);maximum=self.chunk;batch=self.batch;width=self.config['hidden_size']
        if count<1 or any(slot+count>self.capacity for slot in self.slots):raise ValueError('invalid prefill length/capacity')
        if len(positions)!=batch or any(len(row)!=count or any(len(p)!=3 for p in row) for row in positions):raise ValueError('invalid prefill THW positions')
        self.prefill_compile_ms=0;self.prefill_inference_ms=0;self.prefill_chunk_plan=[];offset=0;profiles={}
        while offset<count:
            chunk=self._prefill_chunk_size(count-offset,maximum)
            self.prefill_chunk_plan.append(chunk)
            if chunk not in profiles:profiles[chunk]=self._resident_prefill_graphs(chunk)
            offset+=chunk
        self.poisoned=True;prediction=None;self.prefill_profile=dict(prepare_ms=0,encode_ms=0,gpu_ms=0,wait_ms=0)
        if self.joined_prefill_enabled:self.prefill_profile['joined_prefill_pairs']=0
        if self.packed_prefill_norm_enabled:self.prefill_profile['packed_prefill_norm_dispatches']=0
        if self.global_half_alpha_enabled:self.prefill_profile['global_half_alpha_dispatches']=0
        if 'mlp_prefill_graph' in garnet.backend_info():self.prefill_profile['mlp_prefill_graph_dispatches']=0
        if 'qwen35_wide_batch_prefill' in garnet.backend_info():self.prefill_profile['wide_batch_prefill_dispatches']=0
        if 'prefill_stream' in garnet.backend_info():self.prefill_profile['prefill_partial_submissions']=0
        if 'deferred_state_zero' in garnet.backend_info():
            self.prefill_profile['deferred_zero_bytes']=0;self.prefill_profile['deferred_zero_states']=0
        if self.joined_global_mlp_enabled:self.prefill_profile['joined_global_mlp_blocks']=0
        offset=0
        for chunk in self.prefill_chunk_plan:
            graphs=profiles[chunk]
            values=hidden_tokens[offset:offset+chunk];length=len(values)
            while len(values)<chunk:values.append(self.io.zeros([batch,width]))
            hidden=self.io.stack_tokens(values)
            flag=tensor.tensor([1 if i<length else 0 for b in range(batch) for i in range(chunk)],shape=[batch,chunk],dtype=tensor.float32)
            pos=[v for row in positions for i in range(chunk) for v in (row[offset+i] if i<length else [0,0,0])]
            position=tensor.tensor(pos,shape=[batch,chunk,3],dtype=tensor.float32)
            slots=tensor.tensor([slot+offset+min(i,length) for slot in self.slots for i in range(chunk)],shape=[batch,chunk],dtype=tensor.float32)
            prediction=self.session.run(graphs,hidden,position,slots,flag,length-1 if predict and offset+length==count else -1)
            self.prefill_inference_ms+=self.session.last_inference_ms
            for key in self.prefill_profile:self.prefill_profile[key]+=getattr(self.session,key)
            offset+=length
        for b in range(batch):self.slots[b]+=count
        self.poisoned=False
        return prediction if predict else None

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
        try:self._build_graph(graph,forward(*symbols),symbols,checkpoint)
        except:graph.close();raise
        return graph

    def close(self):
        if getattr(self,'session',None) is not None:self.session.close();self.session=None
        if hasattr(self,'prefill_graphs'):
            for graph in self.prefill_graphs.values():graph.close()
            self.prefill_graphs={}
        CPUDecoder.close(self)
        self.io.close()
