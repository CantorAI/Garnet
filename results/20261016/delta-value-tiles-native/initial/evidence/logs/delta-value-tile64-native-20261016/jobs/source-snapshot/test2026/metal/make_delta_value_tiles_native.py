"""Private, feature-gated width64 integration for confirmed chunk80 geometry only."""
from make_delta_value_tiles import kernel

def replace(source,old,new,count=1):
 assert source.count(old)==count,(old,source.count(old),count);return source.replace(old,new)

def context(source):
 selected=kernel(source,64)
 source=replace(source,'id<MTLComputePipelineState> deltaSplitParallel,deltaChunk;','id<MTLComputePipelineState> deltaSplitParallel,deltaChunk,deltaChunkValue64;')
 source=replace(source,'size_t visionPrefetchDispatches=0,visionVectorDispatches=0;','size_t visionPrefetchDispatches=0,visionVectorDispatches=0,deltaValueTile64Dispatches=0;')
 append='''#ifdef GARNET_METAL_DELTA_VALUE_TILE64
        source=[source stringByAppendingString:[NSString stringWithUTF8String:R"METAL(
'''+selected+'''
)METAL"]];
#endif
        MTLCompileOptions* compileOptions=nil;'''
 source=replace(source,'MTLCompileOptions* compileOptions=nil;',append)
 needle='if(!impl_->kernels->deltaChunk)throw std::runtime_error("Cannot create recurrent chunk pipeline");'
 source=replace(source,needle,needle+'''
#ifdef GARNET_METAL_DELTA_VALUE_TILE64
        impl_->kernels->deltaChunkValue64=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gated_delta_chunk_value64"] error:&error];
        if(!impl_->kernels->deltaChunkValue64)throw std::runtime_error("Cannot create width64 recurrent chunk pipeline");
#endif''')
 needle='size_t MetalContext::LastVisionVectorDispatches() const {return impl_->visionVectorDispatches;}'
 source=replace(source,needle,needle+'\nsize_t MetalContext::LastDeltaValueTile64Dispatches() const {return impl_->deltaValueTile64Dispatches;}')
 source=replace(source,'impl_->visionVectorDispatches=0;','impl_->visionVectorDispatches=0;impl_->deltaValueTile64Dispatches=0;',2)
 needle='auto pipeline=pipelines[operation];'
 source=replace(source,needle,needle+'''
#ifdef GARNET_METAL_DELTA_VALUE_TILE64
    if(operation==19&&parameterBytes==10*sizeof(uint64_t)&&inputs.size()==3&&secondOutput){
        const auto* fields=static_cast<const uint64_t*>(parameters);
        auto candidate=impl_->kernels->deltaChunkValue64;
        bool selected=(fields[0]==1||fields[0]==2)&&fields[1]==80&&fields[2]==16&&fields[3]==128&&fields[4]==128&&
            fields[5]%4==0&&fields[6]%4==0&&fields[8]%4==0&&fields[9]%4==0&&
            threads==fields[0]*fields[2]*(fields[4]/16)&&threads%4==0&&
            c.dataType==MPSDataTypeFloat32&&c.offset==0&&
            secondOutput->impl_->matrix.dataType==MPSDataTypeFloat32&&secondOutput->impl_->matrix.offset==0;
        for(const auto& input:inputs)selected=selected&&input.impl_->matrix.dataType==MPSDataTypeFloat32&&input.impl_->matrix.offset==0;
        if(selected&&candidate&&candidate.threadExecutionWidth==32&&candidate.maxTotalThreadsPerThreadgroup>=128&&
            candidate.staticThreadgroupMemoryLength<=impl_->device.maxThreadgroupMemoryLength){
            pipeline=candidate;threads/=4;++impl_->deltaValueTile64Dispatches;
        }
    }
#endif''')
 return source

def header(source):
 needle='size_t LastVisionVectorDispatches() const;';return replace(source,needle,needle+'\n    size_t LastDeltaValueTile64Dispatches() const;')

def graph(source):
 source=replace(source,'visionVectorDispatches_=0;mlpGraphDispatches_=0;','visionVectorDispatches_=0;mlpGraphDispatches_=0;deltaValueTile64Dispatches_=0;',2)
 needle='visionVectorDispatches_=gpu_->LastVisionVectorDispatches();';return replace(source,needle,needle+'\n        deltaValueTile64Dispatches_=gpu_->LastDeltaValueTile64Dispatches();')

def graph_header(source):
 source=replace(source,'size_t visionPrefetchDispatches_=0,visionVectorDispatches_=0,mlpGraphDispatches_=0;','size_t visionPrefetchDispatches_=0,visionVectorDispatches_=0,mlpGraphDispatches_=0,deltaValueTile64Dispatches_=0;')
 needle='size_t LastVisionVectorDispatches() const {return visionVectorDispatches_;}';source=replace(source,needle,needle+'\n    size_t LastDeltaValueTile64Dispatches() const {return deltaValueTile64Dispatches_;}')
 needle='APISET().AddProp("vision_vector_dispatches",&MetalCompiledGraph::LastVisionVectorDispatches);';return replace(source,needle,needle+'\n        APISET().AddProp("delta_value_tile64_dispatches",&MetalCompiledGraph::LastDeltaValueTile64Dispatches);')

def session(source):
 source=replace(source,'joinedGlobalMlpBlocks_=0;','joinedGlobalMlpBlocks_=0;deltaValueTile64Dispatches_=0;',4)
 needle='joinedGlobalMlpBlocks_=gpu_->LastJoinedGlobalMlpBlocks();';return replace(source,needle,needle+'deltaValueTile64Dispatches_=gpu_->LastDeltaValueTile64Dispatches();')

def session_header(source):
 source=replace(source,'size_t joinedGlobalMlpBlocks_=0;','size_t joinedGlobalMlpBlocks_=0,deltaValueTile64Dispatches_=0;')
 needle='size_t LastJoinedGlobalMlpBlocks() const {return joinedGlobalMlpBlocks_;}';source=replace(source,needle,needle+'\n    size_t LastDeltaValueTile64Dispatches() const {return deltaValueTile64Dispatches_;}')
 needle='APISET().AddProp("joined_global_mlp_blocks",&MetalDecoderSession::LastJoinedGlobalMlpBlocks);';return replace(source,needle,needle+'\n        APISET().AddProp("delta_value_tile64_dispatches",&MetalDecoderSession::LastDeltaValueTile64Dispatches);')

def cmake(source):
 text='''option(GARNET_METAL_DELTA_VALUE_TILE64 "Test original-bit width64 recurrent chunk80 on confirmed geometry" OFF)
if(GARNET_METAL_DELTA_VALUE_TILE64)
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_DELTA_VALUE_TILE64=1)
endif()
''';return replace(source,'option(GARNET_METAL_JOINED_GLOBAL_MLP ',text+'option(GARNET_METAL_JOINED_GLOBAL_MLP ')

def entry(source):
 needle='#ifdef GARNET_METAL_JOINED_GLOBAL_MLP';return replace(source,needle,'#ifdef GARNET_METAL_DELTA_VALUE_TILE64\n        info+="; delta_value_tile64";\n#endif\n'+needle)

def decoder(source):
 needle="if self.global_half_alpha_enabled:self.prefill_profile['global_half_alpha_dispatches']=0";return replace(source,needle,needle+"\n        if 'delta_value_tile64' in garnet.backend_info():self.prefill_profile['delta_value_tile64_dispatches']=0")

def worker(source):
 needle="mlp_graph_enabled='mlp_prefill_graph' in garnet.backend_info()";source=replace(source,needle,needle+";delta_tile64_enabled='delta_value_tile64' in garnet.backend_info()")
 needle="mlp_graph_counts=[decoder.prefill_profile.get('mlp_prefill_graph_dispatches',0) if mlp_graph_enabled else 0]";source=replace(source,needle,needle+";delta_tile64_counts=[decoder.prefill_profile.get('delta_value_tile64_dispatches',0) if delta_tile64_enabled else 0]")
 needle='mlp_graph_counts.append(decoder.session.mlp_prefill_graph_dispatches if mlp_graph_enabled else 0)';source=replace(source,needle,needle+';delta_tile64_counts.append(decoder.session.delta_value_tile64_dispatches if delta_tile64_enabled else 0)')
 needle='mlp_prefill_graph_enabled=mlp_graph_enabled,mlp_prefill_graph_counts=mlp_graph_counts,';return replace(source,needle,needle+'\n            delta_value_tile64_enabled=delta_tile64_enabled,delta_value_tile64_counts=delta_tile64_counts,')

def chunk_reference(source):
 needle='  h=session.hidden().tolist();state=session.snapshot(0);a=state[0].tolist();b=state[1].tolist();resident=[]'
 code='''  if 'delta_value_tile64' in garnet.backend_info():
   expected_tile=int(kind=='linear_attention' and len(sys.argv)>5 and sys.argv[5]=='linear-chunk' and batch in (1,2) and chunk==80 and c['linear_num_value_heads']==16 and c['linear_key_head_dim']==128 and c['linear_value_head_dim']==128)
   assert session.delta_value_tile64_dispatches==expected_tile,'Resident delta value tile route mismatch'
   assert graph.delta_value_tile64_dispatches==expected_tile,'Explicit delta value tile route mismatch'
   print('delta_value_tile64_trained_route '+json.dumps(dict(kind=kind,layer=idx,batch=batch,chunk=chunk,expected=expected_tile,session=session.delta_value_tile64_dispatches,graph=graph.delta_value_tile64_dispatches)),flush=True)
''';return replace(source,needle,code+needle)

OPERATIONS={'src/backends/metal/metal_context.mm':context,'src/backends/metal/metal_context.h':header,
 'src/backends/metal/compiled_graph.cpp':graph,'src/backends/metal/compiled_graph.h':graph_header,
 'src/backends/metal/decoder_session.cpp':session,'src/backends/metal/decoder_session.h':session_header,
 'cmake/GarnetMetal.cmake':cmake,'src/entry/metal_main.cpp':entry,'xModel/qwen35/vl_0_8b/metal_decoder.py':decoder,
 'test2026/metal/frame_worker_garnet.py':worker,'test2026/metal/run_matched_garnet.py':worker,'test2026/metal/run_chunk_graph.py':chunk_reference}
