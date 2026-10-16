"""Private decode-only graph/session pairing after complete repeated component selection."""
import make_paired_norm_rope as component
MODELS={'Qwen3-0.6B':(16,8,128,True,0),'Qwen2.5-Coder-0.5B-Instruct':(14,2,64,False,0),'Qwen3.5-0.8B':(8,2,256,True,1)}
def replace(source,old,new,count=1):
 assert source.count(old)==count,(old,source.count(old),count);return source.replace(old,new)
def graph(source,models):
 source=replace(source,'visionVectorDispatches_=0;mlpGraphDispatches_=0;','visionVectorDispatches_=0;mlpGraphDispatches_=0;pairedNormRopeDispatches_=0;',2)
 source=replace(source,'mlpGraphDispatches_=gpu_->LastMlpGraphDispatches();','mlpGraphDispatches_=gpu_->LastMlpGraphDispatches();pairedNormRopeDispatches_=gpu_->LastPairedNormRopeDispatches();')
 choices=[]
 for name in sorted(models):
  q,k,dim,normalize,offset=MODELS[name]
  choices.append(f'(op.dimensions[0]=={q}&&next.dimensions[0]=={k}&&op.dimensions[1]=={dim}&&op.dimensions[3]=={int(normalize)}&&op.offset=={offset})')
 code='''#ifdef GARNET_METAL_PAIRED_NORM_ROPE
        if(!profileOps&&op.kind==Kind::NormRope&&i+1<ops_.size()&&op.dimensions.size()==4&&
            (slots_[op.a].rows==1||slots_[op.a].rows==2)&&(packedOutput||residentOps_[i+1])){
            const auto& next=ops_[i+1];
            if(next.kind==Kind::NormRope&&next.dimensions.size()==4&&op.arguments.size()==4&&next.arguments.size()==4&&
                op.arguments[2]==next.arguments[2]&&op.arguments[3]==next.arguments[3]&&
                op.dimensions[1]==next.dimensions[1]&&op.dimensions[2]==next.dimensions[2]&&op.dimensions[3]==next.dimensions[3]&&
                op.epsilon==next.epsilon&&op.offset==next.offset&&op.out!=next.out&&
                (POLICY)){
                if(gpu.EncodePairedNormRope(a,slots_[next.a].matrix,b,slots_[next.b].matrix,
                    slots_[op.arguments[2]].matrix,slots_[op.arguments[3]].matrix,out,slots_[next.out].matrix,
                    op.dimensions[0],next.dimensions[0],op.dimensions[1],op.dimensions[2],op.epsilon,op.offset,bool(op.dimensions[3]))){++i;continue;}
            }
        }
#endif
'''.replace('POLICY','||'.join(choices))
 return replace(source,'#ifdef GARNET_METAL_MLP_PREFILL_GRAPH',code+'#ifdef GARNET_METAL_MLP_PREFILL_GRAPH')
def graph_header(source):
 source=replace(source,'size_t visionPrefetchDispatches_=0,visionVectorDispatches_=0,mlpGraphDispatches_=0;','size_t visionPrefetchDispatches_=0,visionVectorDispatches_=0,mlpGraphDispatches_=0,pairedNormRopeDispatches_=0;')
 needle='size_t LastMlpGraphDispatches() const {return mlpGraphDispatches_;}';source=replace(source,needle,needle+'\n    size_t LastPairedNormRopeDispatches() const {return pairedNormRopeDispatches_;}')
 needle='APISET().AddProp("mlp_prefill_graph_dispatches",&MetalCompiledGraph::LastMlpGraphDispatches);'
 return replace(source,needle,needle+'\n        APISET().AddProp("paired_norm_rope_dispatches",&MetalCompiledGraph::LastPairedNormRopeDispatches);')
def session(source):
 source=replace(source,'joinedGlobalMlpBlocks_=0;','joinedGlobalMlpBlocks_=0;pairedNormRopeDispatches_=0;',4)
 return replace(source,'joinedGlobalMlpBlocks_=gpu_->LastJoinedGlobalMlpBlocks();','joinedGlobalMlpBlocks_=gpu_->LastJoinedGlobalMlpBlocks();pairedNormRopeDispatches_=gpu_->LastPairedNormRopeDispatches();')
def session_header(source):
 source=replace(source,'size_t joinedGlobalMlpBlocks_=0;','size_t joinedGlobalMlpBlocks_=0,pairedNormRopeDispatches_=0;')
 needle='size_t LastJoinedGlobalMlpBlocks() const {return joinedGlobalMlpBlocks_;}';source=replace(source,needle,needle+'\n    size_t LastPairedNormRopeDispatches() const {return pairedNormRopeDispatches_;}')
 needle='APISET().AddProp("joined_global_mlp_blocks",&MetalDecoderSession::LastJoinedGlobalMlpBlocks);'
 return replace(source,needle,needle+'\n        APISET().AddProp("paired_norm_rope_dispatches",&MetalDecoderSession::LastPairedNormRopeDispatches);')
def cmake(source):
 text='''option(GARNET_METAL_PAIRED_NORM_ROPE "Test original-bit paired Q/K norm-RoPE on confirmed decode geometry" OFF)
if(GARNET_METAL_PAIRED_NORM_ROPE)
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PAIRED_NORM_ROPE=1)
endif()
'''
 return replace(source,'option(GARNET_METAL_JOINED_GLOBAL_MLP ',text+'option(GARNET_METAL_JOINED_GLOBAL_MLP ')
def entry(source):
 needle='#ifdef GARNET_METAL_JOINED_GLOBAL_MLP';return replace(source,needle,'#ifdef GARNET_METAL_PAIRED_NORM_ROPE\n        info+="; paired_norm_rope";\n#endif\n'+needle)
def decoder(source):
 needle="if self.global_half_alpha_enabled:self.prefill_profile['global_half_alpha_dispatches']=0"
 return replace(source,needle,needle+"\n        if 'paired_norm_rope' in garnet.backend_info():self.prefill_profile['paired_norm_rope_dispatches']=0")
def worker(source):
 needle="mlp_graph_enabled='mlp_prefill_graph' in garnet.backend_info()";source=replace(source,needle,needle+";paired_rope_enabled='paired_norm_rope' in garnet.backend_info()")
 needle="mlp_graph_counts=[decoder.prefill_profile.get('mlp_prefill_graph_dispatches',0) if mlp_graph_enabled else 0]";source=replace(source,needle,needle+";paired_rope_counts=[decoder.prefill_profile.get('paired_norm_rope_dispatches',0) if paired_rope_enabled else 0]")
 needle='mlp_graph_counts.append(decoder.session.mlp_prefill_graph_dispatches if mlp_graph_enabled else 0)';source=replace(source,needle,needle+';paired_rope_counts.append(decoder.session.paired_norm_rope_dispatches if paired_rope_enabled else 0)')
 needle='mlp_prefill_graph_enabled=mlp_graph_enabled,mlp_prefill_graph_counts=mlp_graph_counts,'
 return replace(source,needle,needle+'\n            paired_norm_rope_enabled=paired_rope_enabled,paired_norm_rope_counts=paired_rope_counts,')
def dense_reference(source,models):
 selected=[MODELS[n][:3] for n in sorted(models) if n!='Qwen3.5-0.8B']
 needle='  h=session.hidden().tolist();state=session.snapshot(0);a=state[0].tolist();b=state[1].tolist();resident=[]'
 code='''  if 'paired_norm_rope' in garnet.backend_info():
   expected_rope=int(batch*chunk in (1,2) and (c['num_attention_heads'],c['num_key_value_heads'],c['head_dim']) in SELECTED)
   assert session.paired_norm_rope_dispatches==expected_rope,'Resident paired norm-RoPE route mismatch'
   assert graph.paired_norm_rope_dispatches==expected_rope,'Explicit paired norm-RoPE route mismatch'
   print('paired_norm_rope_trained_route '+json.dumps(dict(batch=batch,chunk=chunk,expected=expected_rope,session=session.paired_norm_rope_dispatches,graph=graph.paired_norm_rope_dispatches)),flush=True)
'''.replace('SELECTED',repr(selected))
 return replace(source,needle,code+needle)
def operations(models):
 assert models and len(models)==len(set(models)) and set(models)<=set(MODELS)
 return {'src/backends/metal/metal_context.mm':component.context,'src/backends/metal/metal_context.h':component.header,
 'src/backends/metal/compiled_graph.cpp':lambda source:graph(source,models),'src/backends/metal/compiled_graph.h':graph_header,
 'src/backends/metal/decoder_session.cpp':session,'src/backends/metal/decoder_session.h':session_header,
 'cmake/GarnetMetal.cmake':cmake,'src/entry/metal_main.cpp':entry,'xModel/qwen35/vl_0_8b/metal_decoder.py':decoder,
 'test2026/metal/frame_worker_garnet.py':worker,'test2026/metal/run_matched_garnet.py':worker,
 'test2026/metal/run_dense_reference.py':lambda source:dense_reference(source,models)}
