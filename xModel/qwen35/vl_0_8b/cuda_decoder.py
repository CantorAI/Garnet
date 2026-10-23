# SPDX-License-Identifier: Apache-2.0
"""CUDA orchestration over the shared Qwen3.5 Tensor Expression programs.

One engine/weight set per decoder layer serves prefill and decode. Request
states, hidden rows, metadata and engine outputs stay on the GPU. The initial
serial prefill is a correctness path; chunked prefill is a separate optimization.
"""
import garnet
import tensor
import json
from .decoder_layer import T,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer,norm
from .vision_encoder import Qwen35VisionEncoder

class CUDADecoder:
    def __init__(self,checkpoint,config,batch=1,capacity=384,chunk_size=1,fused_delta=False,fused_attention=False,projection_workspace_bytes=0,retain_linear_inputs=False,external_weights=False,shared_io=None,shared_decoder=None,register_delta_128=False):
        if batch not in (1,2,4,8) or capacity<32 or capacity>8192:
            raise ValueError('CUDA requires a batch1/2/4/8 profile and capacity32..8192')
        if chunk_size not in (1,4,8,16,32,64,128):raise ValueError('CUDA chunk profile must be1/4/8/16/32/64/128')
        if chunk_size>16 and (not fused_delta or not fused_attention):raise ValueError('Wide chunks require both fused cores')
        if external_weights and chunk_size==1:raise ValueError('External projection inputs require chunked programs')
        if fused_delta and chunk_size==1:raise ValueError('Fused delta requires a chunk profile')
        if fused_attention and chunk_size==1:raise ValueError('Fused attention requires a chunk profile')
        if retain_linear_inputs and (not fused_delta or chunk_size==1):raise ValueError('Retained projections require a fused delta chunk')
        if register_delta_128 and (not fused_delta or config['linear_key_head_dim']!=128 or config['linear_value_head_dim']!=128):raise ValueError('Register delta requires fused128 dimensions')
        self.register_delta_128=register_delta_128
        self.retain_linear_inputs=retain_linear_inputs
        self.fused_delta=fused_delta
        self.fused_attention=fused_attention
        if projection_workspace_bytes not in (0,16384):raise ValueError('Projection workspace must be0 or16384bytes')
        self.projection_workspace_bytes=projection_workspace_bytes
        self.config=config;self.batch=batch;self.capacity=capacity;self.chunk_size=chunk_size;self.layers=[];self.head=None
        self.checkpoint=str(checkpoint);self.external_weights=external_weights;self.owns_io=shared_io is None
        self.io=shared_io if shared_io is not None else garnet.cuda_tensor_io()
        if self.owns_io:self.io.open(checkpoint)
        if shared_decoder is not None:
            if shared_decoder.io is not self.io or shared_decoder.config!=config or shared_decoder.batch!=batch or shared_decoder.capacity!=capacity or shared_decoder.checkpoint!=self.checkpoint or shared_decoder.head is None or any(shared_decoder.slots):
                raise ValueError('Shared decoder state requires the same ready, idle checkpoint/batch/capacity/IO')
        self.hidden=self.io.zeros([batch,config['hidden_size']] if chunk_size==1 else [batch,chunk_size,config['hidden_size']])
        self.last_hidden=shared_decoder.last_hidden if shared_decoder is not None else self.io.zeros([batch,config['hidden_size']])
        self.active=self.io.zeros([batch] if chunk_size==1 else [batch,chunk_size])
        self.position=self.io.zeros([batch,3] if chunk_size==1 else [batch,chunk_size,3])
        self.slot_tensor=self.io.zeros([batch] if chunk_size==1 else [batch,chunk_size])
        try:
            for idx,kind in enumerate(config['layer_types']):
                if kind=='linear_attention':
                    channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
                    shapes=[[batch,config['hidden_size']],[batch,channels,config['linear_conv_kernel_dim']],
                        [batch,config['linear_num_value_heads'],config['linear_key_head_dim'],config['linear_value_head_dim']],[batch]]
                elif kind=='full_attention':
                    cache=[batch,config['num_key_value_heads'],capacity,config['head_dim']]
                    shapes=[[batch,config['hidden_size']],cache,cache,[batch,3],[batch],[batch]]
                else:raise ValueError('unsupported decoder layer '+kind)
                if chunk_size>1:
                    shapes[0]=[batch,chunk_size,config['hidden_size']]
                    if kind=='linear_attention':shapes[3]=[batch,chunk_size]
                    else:shapes[3:]=[[batch,chunk_size,3],[batch,chunk_size],[batch,chunk_size]]
                graph,weight_inputs=self._compile(checkpoint,idx,kind,shapes)
                size=1
                for d in shapes[1][1:]:size*=d
                if shared_decoder is not None:
                    shared=shared_decoder.layers[idx]
                    if shared['kind']!=kind or shared['shapes'][1:3]!=shapes[1:3]:graph.close();raise ValueError('Shared state shape/kind mismatch')
                    state1=shared['state1'];state2=shared['state2']
                else:state1=self.io.zeros(shapes[1]);state2=self.io.zeros(shapes[2])
                self.layers.append(dict(graph=graph,kind=kind,shapes=shapes,state1=state1,state2=state2,state1_width=size,weight_inputs=weight_inputs))
                if retain_linear_inputs and kind=='linear_attention':
                    state2_width=1
                    for d in shapes[2][1:]:state2_width*=d
                    payload=chunk_size*(channels+config['linear_num_value_heads']*config['linear_value_head_dim']+2*config['linear_num_value_heads'])
                    self.layers[-1]['packed_shape']=[batch,chunk_size*config['hidden_size']+size+state2_width+payload]
                print('compiled CUDA Qwen3.5 layer',idx,kind,'ms',graph.compile_ms,flush=True)
            symbol=tensor.input('cuda_final_hidden',shape=[batch,config['hidden_size']],dtype=tensor.float32)
            @T.fusion(name='qwen35_cuda_final_norm',boundary='required')
            def head(hidden):return norm(hidden,'model.language_model.norm',config)
            self.head=garnet.compiled_graph();self.head.build(head(symbol),[symbol],checkpoint)
            self.reset()
        except:
            self.close();raise

    def _compile(self,checkpoint,idx,kind,shapes):
        config=self.config;capacity=self.capacity
        symbols=[tensor.input('cuda_layer_'+str(idx)+'_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
        weight_names=[];weight_inputs=[]
        if self.external_weights:
            p='model.language_model.layers.'+str(idx)+'.'
            suffixes=['mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']
            suffixes+=['linear_attn.'+name+'.weight' for name in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj')] if kind=='linear_attention' else ['self_attn.'+name+'.weight' for name in ('q_proj','k_proj','v_proj','o_proj')]
            for i,suffix in enumerate(suffixes):
                name=p+suffix;info=json.loads(self.io.weight_info(name))
                if info['dtype']!='BF16' or len(info['shape'])!=2:raise ValueError('External projection requires BF16 matrix metadata')
                symbols.append(tensor.input('cuda_layer_'+str(idx)+'_weight_'+str(i),shape=info['shape'],dtype=tensor.bfloat16))
                weight_names.append(name);weight_inputs.append(self.io.weight(name))
        @T.fusion(name='qwen35_cuda_decoder_layer',boundary='required')
        def forward(*args):
            weights={weight_names[i]:value for i,value in enumerate(args[len(shapes):])} if self.external_weights else None
            if self.chunk_size>1:
                from .cuda_chunk import Qwen35CUDAChunk
                if kind=='linear_attention':return Qwen35CUDAChunk(args[0],args[1],args[2],args[3],None,None,config,idx,kind,self.fused_delta,projection_workspace_bytes=self.projection_workspace_bytes,retain_linear_inputs=self.retain_linear_inputs,external_weights=weights,register_delta_128=self.register_delta_128)
                return Qwen35CUDAChunk(args[0],args[1],args[2],args[5],args[3],args[4],config,idx,kind,fused_attention=self.fused_attention,projection_workspace_bytes=self.projection_workspace_bytes,external_weights=weights)
            if kind=='linear_attention':return Qwen35LinearDecoderLayer(*args,config,idx)
            return Qwen35FullDecoderLayer(*args,config,idx,capacity)
        graph=garnet.compiled_graph()
        # Official 4B batch8 builds exceeded both 64MiB and 128MiB caps;
        # an isolated layer build succeeds at 256MiB. This is a limit,
        # not a permanent allocation; retained runtime usage is measured.
        if kind=='full_attention' and self.batch==8:graph.set_builder_workspace_bytes(256<<20)
        # Official4B batch4/context512 exceeds64MiB by10,368 bytes;
        # an isolated same-layer build succeeds with128MiB. Other admitted
        # profiles retain their measured limits and arithmetic unchanged.
        if kind=='full_attention' and self.batch==4 and config['hidden_size']==2560 and capacity==512 and self.chunk_size>=4:
            graph.set_builder_workspace_bytes(128<<20)
        try:graph.build(forward(*symbols),symbols,checkpoint)
        except:graph.close();raise
        return graph,weight_inputs

    def reset(self):
        self.slots=[0 for b in range(self.batch)];self.poisoned=False
        self.io.zero(self.hidden);self.io.zero(self.last_hidden)
        for layer in self.layers:
            self.io.zero(layer['state1']);self.io.zero(layer['state2']);layer.pop('last_packed',None)

    def enable_cuda_graph(self,enabled):
        for layer in self.layers:layer['graph'].enable_cuda_graph(enabled)
        self.head.enable_cuda_graph(enabled)

    def memory_info(self):
        graphs=[json.loads(layer['graph'].stats_json()) for layer in self.layers]+[json.loads(self.head.stats_json())]
        io=json.loads(self.io.memory_json())
        state_bytes=sum(self.io.tensor_bytes(layer['state1'])+self.io.tensor_bytes(layer['state2']) for layer in self.layers)
        request_bytes=sum(self.io.tensor_bytes(t) for t in [self.hidden,self.last_hidden,self.active,self.position,self.slot_tensor])
        output_bytes=sum(g['device_output_bytes'] for g in graphs)
        tracked=all(g['allocator_tracking_enabled'] for g in graphs)
        runtime_bytes=sum(g['trt_allocator_live_bytes'] for g in graphs) if tracked else None
        proposal_bytes=io.get('candidate_workspace_bytes',0)+io.get('candidate_index_bytes',0)+io.get('indexed_head_buffers_bytes',0)
        pool_bytes=io.get('projection_weight_pool_bytes',0)
        weight_bytes=io['embedding_table_bytes']+io['untied_head_bytes']
        native_head_bytes=io.get('native_head_workspace_bytes',0)
        owned=state_bytes+request_bytes+output_bytes+weight_bytes+native_head_bytes+io['head_input_bytes']+io['head_output_bytes']+proposal_bytes+pool_bytes
        return {'graphs':graphs,'register_delta_128':self.register_delta_128,'state_bytes':state_bytes,'request_buffer_bytes':request_bytes,'engine_output_bytes':output_bytes,
            'embedding_head_shared_bytes':io['embedding_head_shared_bytes'],'embedding_table_bytes':io['embedding_table_bytes'],
            'untied_head_bytes':io['untied_head_bytes'],'native_head_workspace_bytes':native_head_bytes,'head_input_bytes':io['head_input_bytes'],'head_output_bytes':io['head_output_bytes'],
            'proposal_io_bytes':proposal_bytes,'proposal_io':{key:io.get(key,0) for key in ('candidate_workspace_bytes','candidate_index_bytes','indexed_head_buffers_bytes')},
            'projection_weight_pool_bytes':pool_bytes,'projection_weight_pool_entries':io.get('projection_weight_pool_entries',0),
            'trt_runtime_allocator_live_bytes':runtime_bytes,'owned_decoder_bytes':owned+runtime_bytes if tracked else None,
            'scope':'Current retained tensors plus tracked TRT runtime allocations; excludes transient tensors, builder workspace, driver and external libraries. Per-engine peaks must not be summed as a process peak.'}

    def step(self,tokens,active=None,positions=None,predict=True):
        if len(tokens)!=self.batch:raise ValueError('CUDA token batch mismatch')
        if self.chunk_size>1 and getattr(self,'resident_controls',False):
            if active is None:active=[1]*self.batch
            if positions is None:positions=[[slot]*3 for slot in self.slots]
            return self.chunk_tokens([tokens],[active],[[p] for p in positions],predict)
        return self.step_hidden(self.io.embedding(tokens),active,positions,predict)

    def chunk_tokens(self,tokens,active,positions,predict=True):
        if getattr(self,'resident_controls',False):return self.chunk_hidden(tokens,active,positions,predict,_token_ids=True)
        return self.chunk_hidden([self.io.embedding(list(row)) for row in tokens],active,positions,predict)

    def step_hidden(self,hidden,active=None,positions=None,predict=True):
        if self.chunk_size>1:
            if active is None:active=[1]*self.batch
            if positions is None:positions=[[self.slots[b]]*3 for b in range(self.batch)]
            return self.chunk_hidden([hidden],[active],[[positions[b]] for b in range(self.batch)],predict)
        if self.poisoned:raise ValueError('reset CUDA request after a failed step')
        if active is None:active=[1 for b in range(self.batch)]
        if len(active)!=self.batch or any(a not in (0,1) for a in active):raise ValueError('invalid CUDA active rows')
        if any(active[b] and self.slots[b]>=self.capacity for b in range(self.batch)):raise ValueError('CUDA context capacity exhausted')
        if positions is None:positions=[[self.slots[b]]*3 for b in range(self.batch)]
        if len(positions)!=self.batch or any(len(p)!=3 for p in positions):raise ValueError('CUDA positions require batch x THW')
        self.poisoned=True;self.io.slice_into(hidden,0,self.hidden)
        self.io.update(self.active,active);self.io.update(self.position,[v for row in positions for v in row]);self.io.update(self.slot_tensor,self.slots)
        self.last_layer_enqueue_ms=[]
        for layer in self.layers:
            if layer['kind']=='linear_attention':inputs=[self.hidden,layer['state1'],layer['state2'],self.active]
            else:inputs=[self.hidden,layer['state1'],layer['state2'],self.position,self.slot_tensor,self.active]
            packed=layer['graph'].run_device(inputs+layer['weight_inputs']);self.last_layer_enqueue_ms.append(layer['graph'].last_inference_ms)
            width=self.config['hidden_size'];self.io.slice_into(packed,0,self.hidden)
            self.io.slice_into(packed,width,layer['state1']);self.io.slice_into(packed,width+layer['state1_width'],layer['state2'])
        self.io.copy_active(self.hidden,self.active,self.last_hidden)
        prediction=self.io.predict(self.head.run_device([self.last_hidden])) if predict else None
        for b in range(self.batch):self.slots[b]+=active[b]
        self.poisoned=False;return prediction

    def chunk_hidden(self,hidden_tokens,active_tokens,positions,predict=True,_token_ids=False):
        if self.poisoned:raise ValueError('reset CUDA request after a failed chunk')
        count=len(hidden_tokens);size=self.chunk_size;batch=self.batch
        if size==1 or count<1 or count>size or len(active_tokens)!=count:
            raise ValueError('invalid CUDA chunk length')
        if len(positions)!=batch or any(len(row)!=count or any(len(p)!=3 for p in row) for row in positions):
            raise ValueError('invalid CUDA chunk positions')
        if any(len(a)!=batch or any(v not in (0,1) for v in a) for a in active_tokens):raise ValueError('invalid CUDA chunk active rows')
        flags=[];slots=[];pos=[];next_slots=list(self.slots)
        for b in range(batch):
            for t in range(size):
                active=active_tokens[t][b] if t<count else 0
                if active and next_slots[b]>=self.capacity:raise ValueError('CUDA context capacity exhausted')
                flags.append(active);slots.append(next_slots[b]);next_slots[b]+=active
                pos.extend(positions[b][t] if t<count else [0,0,0])
        self.poisoned=True
        if _token_ids:self.io.embedding_chunk(hidden_tokens,self.hidden)
        else:
            self.io.zero(self.hidden)
            for t in range(count):self.io.write_chunk_token(hidden_tokens[t],self.hidden,t)
        if getattr(self,'resident_controls',False):self.io.update_request(self.active,self.position,self.slot_tensor,flags,pos,slots)
        else:
            self.io.update(self.active,flags);self.io.update(self.position,pos);self.io.update(self.slot_tensor,slots)
        self.last_layer_enqueue_ms=[]
        for layer in self.layers:
            inputs=[self.hidden,layer['state1'],layer['state2'],self.active] if layer['kind']=='linear_attention' else [self.hidden,layer['state1'],layer['state2'],self.position,self.slot_tensor,self.active]
            packed=layer['graph'].run_device(inputs+layer['weight_inputs']);self.last_layer_enqueue_ms.append(layer['graph'].last_inference_ms)
            width=size*self.config['hidden_size'];self.io.slice_into(packed,0,self.hidden)
            if self.retain_linear_inputs and layer['kind']=='linear_attention':layer['last_packed']=packed
            self.io.slice_into(packed,width,layer['state1']);self.io.slice_into(packed,width+layer['state1_width'],layer['state2'])
        self.io.copy_last_chunk(self.hidden,self.active,self.last_hidden)
        prediction=self.io.predict(self.head.run_device([self.last_hidden])) if predict else None
        self.slots=next_slots;self.poisoned=False;return prediction

    def prefill_sequence(self,hidden_tokens,positions,lengths=None):
        count=len(hidden_tokens)
        if count<1 or len(positions)!=self.batch or any(len(row)!=count for row in positions):raise ValueError('invalid CUDA prefill positions')
        if lengths is None:lengths=[count]*self.batch
        if len(lengths)!=self.batch or any(n<1 or n>count for n in lengths):raise ValueError('invalid CUDA prefill lengths')
        if any(self.slots[b]+lengths[b]>self.capacity for b in range(self.batch)):raise ValueError('CUDA prefill exceeds profile')
        if self.chunk_size>1:
            for start in range(0,count,self.chunk_size):
                end=min(start+self.chunk_size,count)
                prediction=self.chunk_hidden(hidden_tokens[start:end],[[int(i<lengths[b]) for b in range(self.batch)] for i in range(start,end)],
                    [row[start:end] for row in positions],end==count)
            return prediction
        prediction=None
        for i in range(count):
            prediction=self.step_hidden(hidden_tokens[i],[int(i<lengths[b]) for b in range(self.batch)],
                [positions[b][i] for b in range(self.batch)],i+1==count)
        return prediction

    def close(self):
        if self.head is not None:self.head.close();self.head=None
        for layer in self.layers:layer['graph'].close()
        self.layers=[]
        if self.owns_io:self.io.close()
        self.hidden=None;self.last_hidden=None;self.active=None;self.position=None;self.slot_tensor=None

class CUDAImageEncoder:
    def __init__(self,checkpoint,config,gh,gw,batch=1):
        self.shape=[batch,gh*gw,config['in_channels']*config['temporal_patch_size']*config['patch_size']**2]
        symbol=tensor.input('cuda_image_patches',shape=self.shape,dtype=tensor.float32)
        @T.fusion(name='qwen35_cuda_image',boundary='required')
        def forward(patches):return Qwen35VisionEncoder(patches,config,gh,gw)
        self.graph=garnet.compiled_graph()
        try:self.graph.build(forward(symbol),[symbol],checkpoint)
        except:self.graph.close();raise
    def encode_patches(self,patches):return self.graph.run_device([patches])
    def close(self):self.graph.close()
