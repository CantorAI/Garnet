# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 CUDA request orchestration over native Garnet operators.

One static batch/grid profile owns one model. Each image is decoded/encoded
again on every request; no image, prompt, or answer cache is introduced.
Serial prefill is the initial correctness path, not the final speed target.
"""
import json,os,time,threading,hashlib
import garnet

class Model:
    def __init__(self,root,profile,name):
        self.root=root;self.name=name;self.profile=profile;self.vision=None;self.decoder=None;self.grid=None;self.speculative=None
        self.lock=threading.RLock()
        with open(os.path.join(root,'config.json')) as f:self.config=json.loads(f.read())
        if name=='Qwen3.5-2B':
            from xModel.qwen35.vl_2b import model as factory
        elif name=='Qwen3.5-4B':
            from xModel.qwen35.vl_4b import model as factory
        else:raise ValueError('CUDA Qwen3.5 model ID unsupported')
        self.factory=factory;factory.validate_config(self.config)
        self.batch=int(profile.get('batch',1));self.max_input=int(profile.get('maxInputTokens',256))
        self.max_output=int(profile.get('maxOutputTokens',128));self.pixels=int(profile.get('imagePixels',65536))
        if self.batch not in (1,2,4,8) or self.max_input<1 or self.max_output<1 or self.pixels<1024:
            raise ValueError('CUDA batch/capacity profile invalid')
        self.image_min_pixels=profile.get('imageMinPixels')
        if self.image_min_pixels is not None and (type(self.image_min_pixels) is not int or not 1024<=self.image_min_pixels<=self.pixels):raise ValueError('Invalid CUDA minimum image pixels')
        self.capacity=self.max_input+self.max_output
        if self.capacity>8192:raise ValueError('CUDA context profile exceeds admitted capacity')
        self.prefill_chunk=int(profile.get('prefillChunk',1));self.decode_chunk=int(profile.get('decodeChunk',self.prefill_chunk))
        self.shared_projection=bool(profile.get('sharedProjectionWeights',False))
        self.packed_program=bool(profile.get('packedDecoderProgram',False))
        self.pack_prompt=bool(profile.get('packPrompt',False))
        self.join_decode_input_projections=bool(profile.get('joinDecodeInputProjections',False))
        if self.join_decode_input_projections and not (self.packed_program and self.shared_projection and self.decode_chunk==4):
            raise ValueError('Joined decode projections require packed/shared decode width4')
        if self.pack_prompt and not self.packed_program:raise ValueError('Packed prompt inputs require the packed decoder profile')
        self.register_delta_128=bool(profile.get('registerDelta128',False));self.flat_packed_concat=bool(profile.get('flatPackedConcat',False))
        if self.register_delta_128 and (not self.shared_projection or not profile.get('fusedDelta',False) or self.config['text_config']['linear_key_head_dim']!=128 or self.config['text_config']['linear_value_head_dim']!=128):
            raise ValueError('Register delta128 requires the shared fused128 profile')
        if self.flat_packed_concat and not self.packed_program:raise ValueError('Flat packed concatenation requires the packed decoder profile')
        self.resident_controls=bool(profile.get('residentControls',False))
        self.serial_graph_reuse=bool(profile.get('serialGraphReuse',False))
        self.serial_target_graph_reuse=bool(profile.get('serialTargetGraphReuse',False))
        self.shared_output_storage=bool(profile.get('sharePackedOutputStorage',False))
        if self.shared_output_storage and (not self.packed_program or not self.serial_target_graph_reuse):
            raise ValueError('Shared packed output storage requires the packed serial-target profile')
        if self.serial_target_graph_reuse and (not self.packed_program or not self.resident_controls or not profile.get('cudaGraph',False)):
            raise ValueError('Serial target graph reuse requires packed resident CUDA graph profiles')
        if self.serial_graph_reuse and (not self.resident_controls or not profile.get('cudaGraph',False)):
            raise ValueError('Serial graph reuse requires explicit resident controls and CUDA graph profiles')
        if self.resident_controls and self.prefill_chunk<4:raise ValueError('Resident controls require a chunked decoder profile')
        if self.packed_program and not self.shared_projection:raise ValueError('Packed decoder requires the explicit shared projection profile')
        if self.shared_projection:
            if self.prefill_chunk not in (32,64,128) or self.decode_chunk not in (4,8,16) or self.prefill_chunk<=self.decode_chunk or not profile.get('fusedDelta',False) or not profile.get('fusedAttention',False):
                raise ValueError('Shared profiles require wider prefill32/64/128, decode4/8/16 and both fused cores')
        elif self.prefill_chunk!=self.decode_chunk:raise ValueError('Different prefill/decode chunks require the shared projection profile')
        self.speculative_depth=int(profile.get('speculativeMTP',0))
        self.speculative_candidates=int(profile.get('speculativeHeadCandidates',0))
        self.speculative_replay=bool(profile.get('speculativePrefixReplay',False))
        self.deferred_state_commit=bool(profile.get('deferPackedStateCommit',False))
        self.native_prefix_commit=bool(profile.get('nativePackedPrefixCommit',False))
        self.reuse_verified_head=bool(profile.get('reuseVerifiedHead',False))
        if self.reuse_verified_head and not (self.packed_program and self.native_prefix_commit and self.speculative_replay and self.speculative_depth):
            raise ValueError('Verified head reuse requires native packed speculative replay')
        if self.native_prefix_commit and (not self.deferred_state_commit or not self.register_delta_128):
            raise ValueError('Native packed prefix commit requires deferred state and register128 arithmetic')
        if self.deferred_state_commit and (not self.packed_program or not self.speculative_depth or not self.speculative_replay):
            raise ValueError('Deferred canonical state requires packed MTP with prefix replay')
        if self.speculative_replay and (not self.speculative_depth or not profile.get('fusedDelta',False)):
            raise ValueError('Delta prefix replay requires an enabled MTP and fused delta profile')
        if self.speculative_candidates<0 or self.speculative_candidates>min(65536,self.config['text_config']['vocab_size']) or self.speculative_candidates and not self.speculative_depth:
            raise ValueError('Indexed MTP proposal head requires an enabled MTP profile and valid candidate count')
        if self.speculative_depth not in (0,1,3,7) or self.speculative_depth and (self.speculative_depth+1>self.decode_chunk or self.decode_chunk>16):
            raise ValueError('Experimental MTP depth0/1/3/7 must fit the target chunk')
        self.tokenizer=garnet.tokenizer();self.tokenizer.open(root)
        self.stops=[self.tokenizer.token_id('<|im_end|>'),self.tokenizer.token_id('<|endoftext|>')]
        eos=self.config['text_config'].get('eos_token_id')
        if isinstance(eos,list):self.stops.extend(eos)
        elif eos is not None:self.stops.append(eos)
        self.checkpoint=root
        try:
            if self.packed_program:
                from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
                self.decoder=CUDAPackedDecoder(root,self.config['text_config'],self.batch,self.capacity,self.prefill_chunk,self.decode_chunk,
                    int(profile.get('projectionWorkspaceBytes',0)),self.speculative_replay or bool(profile.get('retainLinearInputs',False)),self.register_delta_128,self.flat_packed_concat,self.deferred_state_commit,self.shared_output_storage,self.join_decode_input_projections)
            elif self.shared_projection:
                from xModel.qwen35.vl_0_8b.cuda_hybrid import CUDAHybridDecoder
                self.decoder=CUDAHybridDecoder(root,self.config['text_config'],self.batch,self.capacity,self.prefill_chunk,self.decode_chunk,
                    int(profile.get('projectionWorkspaceBytes',0)),self.speculative_replay or bool(profile.get('retainLinearInputs',False)),self.register_delta_128)
            else:self.decoder=factory.create_cuda_decoder(root,self.config,self.capacity,self.batch,self.prefill_chunk,bool(profile.get('fusedDelta',False)),bool(profile.get('fusedAttention',False)),int(profile.get('projectionWorkspaceBytes',0)),self.speculative_replay or bool(profile.get('retainLinearInputs',False)))
            self.decoder.resident_controls=self.resident_controls
            self.decoder.serial_graph_reuse=self.serial_graph_reuse
            self.decoder.serial_target_graph_reuse=self.serial_target_graph_reuse
            self.decoder.native_prefix_commit=self.native_prefix_commit
            self.decoder.reuse_verified_head=self.reuse_verified_head
            self.decoder.enable_cuda_graph(bool(profile.get('cudaGraph',False)))
            if profile.get('imageGrid'):
                gh,gw=profile['imageGrid'];self._build_vision(int(gh),int(gw))
            if self.speculative_depth:
                from xModel.qwen35.vl_0_8b.cuda_speculative import CUDASpeculativeSession
                self.speculative=CUDASpeculativeSession(self.decoder,self.checkpoint,self.stops,self.speculative_depth,self.speculative_candidates,self.speculative_replay)
        except:
            self.close();raise

    def _build_vision(self,gh,gw):
        if gh<2 or gw<2 or gh%2 or gw%2:raise ValueError('CUDA vision profile requires a positive merged grid')
        if gh*gw*self.config['vision_config']['patch_size']**2>self.pixels:
            raise ValueError('CUDA image grid exceeds loaded pixel profile')
        if self.vision is not None:
            if self.grid!=[gh,gw]:raise ValueError('Image grid differs from loaded CUDA profile; reload the profile explicitly')
            return 0.
        start=time.monotonic();self.vision=self.factory.create_cuda_vision(self.checkpoint,self.config,gh,gw)
        self.grid=[gh,gw];return (time.monotonic()-start)*1000

    def _prepare(self,prompt,image):
        io=self.decoder.io;ids=[];positions=[];visual=None;image_start=-1;visual_count=0;current=0;compile_ms=0.
        def text(value):
            nonlocal current
            for token in self.tokenizer.encode(value):ids.append(token);positions.append([current]*3);current+=1
        text('<|im_start|>user\n')
        if image:
            minimum=getattr(self,'image_min_pixels',None)
            prepared=io.prepare_image(image,self.pixels) if minimum is None else io.prepare_image_profile(image,minimum,self.pixels)
            gh,gw=prepared[1],prepared[2]
            compile_ms=self._build_vision(gh,gw)
            visual=self.vision.encode_patches(prepared[0]);text('<|vision_start|>')
            image_start=len(ids);mh,mw=gh//2,gw//2;visual_count=mh*mw
            for i in range(visual_count):
                ids.append(self.config['image_token_id']);positions.append([current,current+i//mw,current+i%mw])
            current+=max(mh,mw);text('<|vision_end|>')
        text(prompt+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
        if len(ids)>self.max_input:raise ValueError('CUDA prompt exceeds the loaded input profile')
        embeddings=io.embedding(ids)
        if self.pack_prompt:
            # Vision graph output is reused by the next image in this batch.
            # Snapshot each row before another encode can overwrite its tokens.
            if visual is not None and self.batch>1:visual=io.clone(visual)
            return embeddings,positions,ids,current,compile_ms,visual_count,visual,image_start
        hidden=[]
        for i in range(len(ids)):
            if image_start<=i<image_start+visual_count:hidden.append(io.visual_token(visual,i-image_start))
            else:hidden.append(io.row(embeddings,i))
        return hidden,positions,ids,current,compile_ms,visual_count

    def infer_batch(self,prompts,images,limit,on_token=None):
        with self.lock:
            if on_token is not None:
                if not callable(on_token):raise TypeError('on_token must be callable')
                if self.speculative is not None:raise ValueError('Model streaming callback currently requires target-only decoding')
            if len(prompts)!=self.batch or len(images)!=self.batch:raise ValueError('Request batch must match loaded CUDA profile')
            if limit<1 or limit>self.max_output:raise ValueError('Output limit exceeds loaded CUDA profile')
            started=time.monotonic()
            if self.speculative is not None:self.speculative.reset()
            else:self.decoder.reset()
            rows=[self._prepare(prompts[b],images[b]) for b in range(self.batch)]
            lengths=[len(r[2]) for r in rows];count=max(lengths)
            if self.pack_prompt:
                hidden=list(self.decoder.io.pack_prompt([r[0] for r in rows],[r[6] for r in rows],[r[7] for r in rows]))
            else:
                zero=self.decoder.io.zeros([1,self.config['text_config']['hidden_size']]);hidden=[]
                for t in range(count):hidden.append(self.decoder.io.stack_rows([r[0][t] if t<len(r[0]) else zero for r in rows]))
            positions=[r[1]+[[0,0,0]]*(count-len(r[1])) for r in rows]
            input_done=time.monotonic()
            if self.speculative is not None:prediction=self.speculative.prefill(hidden,positions,lengths,[r[3] for r in rows])
            else:prediction=self.decoder.prefill_sequence(hidden,positions,lengths)
            first=time.monotonic();output=[[] for b in range(self.batch)];active=[1]*self.batch
            sampled=[0]*self.batch;finish=['length']*self.batch;stop_tokens=[None]*self.batch;decode_steps=0
            speculative_metrics=None
            if self.speculative is not None:
                generated=self.speculative.generate(limit);output=generated['generated_token_ids']
                sampled=generated['sampled_tokens_including_stop'];finish=generated['finish_reasons'];stop_tokens=generated['stop_token_ids']
                decode_steps=generated['decode_steps'];speculative_metrics=generated['speculative_metrics']
            else:
                current=[r[3] for r in rows]
                for step in range(limit):
                    for b in range(self.batch):
                        if active[b]:
                            sampled[b]+=1
                            # Each prediction is synchronized by the decoder before return.
                            # Include EOS in the sampled stream; never emit inactive rows.
                            if on_token is not None:on_token(b,prediction[b],sampled[b],lengths[b])
                            if prediction[b] in self.stops:
                                active[b]=0;finish[b]='eos';stop_tokens[b]=prediction[b]
                            else:output[b].append(prediction[b])
                    if not any(active) or step+1==limit:break
                    prediction=self.decoder.step(prediction,active,[[p]*3 for p in current])
                    decode_steps+=1
                    for b in range(self.batch):current[b]+=active[b]
            done=time.monotonic()
            return {'status':'ok','model_id':self.name,'backend':'Garnet CUDA/TensorRT Qwen3.5','batch':self.batch,
                'texts':[self.tokenizer.decode(o) for o in output],'generated_token_ids':output,'generated_tokens':[len(o) for o in output],
                'sampled_tokens_including_stop':sampled,'finish_reasons':finish,'stop_token_ids':stop_tokens,'decode_steps':decode_steps,
                'prompt_token_ids':[r[2] for r in rows],'prompt_tokens':lengths,
                'visual_tokens_per_row':[r[5] for r in rows],
                'visual_tokens':rows[0][5] if all(r[5]==rows[0][5] for r in rows) else [r[5] for r in rows],
                'ttft_seconds':first-started,'total_seconds':done-started,'input_seconds':input_done-started,
                'prefill_seconds':first-input_done,'decode_seconds':done-first,'vision_compile_ms':sum(r[4] for r in rows),
                'profile':self.profile,'source':__file__,'cache_policy':{'prefix':False,'image':False,'answer':False},
                'decoder_profiles':{'prefill_chunk':self.prefill_chunk,'decode_chunk':self.decode_chunk,'shared_projection_weights':self.shared_projection,'packed_decoder_program':self.packed_program,'resident_controls':self.resident_controls,'register_delta_128':self.register_delta_128,'flat_packed_concat':self.flat_packed_concat,'serial_target_graph_reuse':self.serial_target_graph_reuse,'deferred_state_commit':self.deferred_state_commit,'native_prefix_commit':self.native_prefix_commit,'shared_output_storage':self.shared_output_storage},
                'speculative':{'enabled':self.speculative is not None,'mtp_depth':self.speculative_depth,
                    'reuse_verified_head':self.reuse_verified_head,
                    'serial_graph_reuse':self.serial_graph_reuse if self.speculative is not None else False,
                    'head_candidates':self.speculative_candidates,'head_method':'indexed shared BF16 table' if self.speculative_candidates else 'full vocabulary',
                    'proposal_source':'native checkpoint MTP' if self.speculative is not None else None,
                    'target_verification':self.speculative is not None,'partial_commit':(('packed_delta_replay_kv_selection' if self.packed_program else 'replay_delta_restore_kv') if self.speculative_replay else 'restore_recompute') if self.speculative is not None else None},
                'speculative_metrics':speculative_metrics,
                'arithmetic':{'weights':'BF16','projection_input_output':'BF16','projection_accumulation':'FP32',
                    'projection_reduced_precision_reduction':False,
                    'full_attention_projection_layout':'chunk_batched',
                    'residual_state_norm_attention':'FP32','equivalent_to_transformers_full_bf16':False},
                'prompt_sha256':[hashlib.sha256(p.encode()).hexdigest() for p in prompts]}

    def infer(self,prompt,image,limit,on_token=None):
        if self.batch!=1:raise ValueError('Use infer_batch with the loaded batch profile')
        r=self.infer_batch([prompt],[image],limit,on_token=on_token);r['text']=r['texts'][0];return r

    def close(self):
        try:
            if self.speculative is not None:self.speculative.close()
        finally:
            self.speculative=None
            try:
                if self.vision is not None:self.vision.close()
            finally:
                self.vision=None
                if self.decoder is not None:self.decoder.close();self.decoder=None

    def runtime_status(self):
        return {'state':'ready' if self.decoder is not None else 'stopped',
            'ready':self.decoder is not None,'backend':'CUDA/TensorRT',
            'precision':'BF16 weights/projections; FP32 residuals/state',
            'frontend':'native XLang3 Qwen3.5','profile_json':json.dumps(self.profile),
            'batch':self.batch}

    def memory_info(self):
        decoder=self.decoder.memory_info()
        vision=json.loads(self.vision.graph.stats_json()) if self.vision is not None else None
        total=decoder['owned_decoder_bytes']
        if vision is not None:
            total=total+vision['trt_allocator_live_bytes']+vision['device_output_bytes'] if total is not None and vision['allocator_tracking_enabled'] else None
        speculative=self.speculative.memory_info() if self.speculative is not None else None
        if speculative is not None:total=total+speculative['owned_speculative_bytes'] if total is not None and speculative['owned_speculative_bytes'] is not None else None
        return {'decoder':decoder,'vision':vision,'speculative':speculative,'owned_model_bytes':total,
            'scope':'Retained tensor bytes plus tracked TRT runtime allocator; transient request/build allocations, driver/external libraries and graph executable overhead may be absent. Not process VRAM peak.'}

    def release_runtime(self):self.close()

    def infer_json(self,prompt,image,limit):
        # Public API accepts one prompt/path at batch1 or explicit lists at
        # the loaded batch. No silent request replication or limit clamping.
        prompts=prompt if isinstance(prompt,list) else [prompt]
        images=image if isinstance(image,list) else [image or '']
        paths=[];temporary=[]
        try:
            for value in images:
                if isinstance(value,bytes):
                    import tempfile
                    handle=tempfile.NamedTemporaryFile(suffix='.jpg',delete=False)
                    temporary.append(handle.name)
                    try:handle.write(value)
                    finally:handle.close()
                    paths.append(handle.name)
                else:paths.append(value or '')
            result=self.infer_batch(prompts,paths,int(limit))
            if self.batch==1:result['text']=result['texts'][0]
            return json.dumps(result)
        finally:
            for path in temporary:os.remove(path)

def create_serving_model(root,profile_json,name):
    return Model(root,json.loads(profile_json),name)
