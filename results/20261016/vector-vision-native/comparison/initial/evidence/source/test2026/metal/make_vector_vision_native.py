"""Private native integration of the audited float4 ordered context kernel."""
def replace(source,old,new,count=1):
    assert source.count(old)==count,(old,source.count(old),count)
    return source.replace(old,new)

def shader(source,selected):
    start=selected.index('kernel void vision_vector4(')
    function=selected[start:]
    assert function.count('kernel void ')==1 and function.count('result=fma(')==9
    return replace(source,')GARNET";',function+'\n)GARNET";')

def context(source):
    source=replace(source,'id<MTLComputePipelineState> visionPrefetch;','id<MTLComputePipelineState> visionPrefetch,visionVector;')
    source=replace(source,'size_t visionPrefetchDispatches=0;','size_t visionPrefetchDispatches=0,visionVectorDispatches=0;')
    old='impl_->kernels->visionPrefetch=[impl_->device newComputePipelineStateWithFunction:[prefetchLibrary newFunctionWithName:@"vision_prefetch8"] error:&error];'
    source=replace(source,old,old+'\n            impl_->kernels->visionVector=[impl_->device newComputePipelineStateWithFunction:[prefetchLibrary newFunctionWithName:@"vision_vector4"] error:&error];')
    old='size_t MetalContext::LastVisionPrefetchDispatches() const {return impl_->visionPrefetchDispatches;}'
    source=replace(source,old,old+'\nsize_t MetalContext::LastVisionVectorDispatches() const {return impl_->visionVectorDispatches;}')
    source=replace(source,'impl_->visionPrefetchDispatches=0;','impl_->visionPrefetchDispatches=0;impl_->visionVectorDispatches=0;',2)
    source=replace(source,'bool serialVisionPrefetch=false;','bool serialVisionPrefetch=false,vectorVisionContext=false;')
    old='pipeline=prefetch;serialVisionPrefetch=true;++impl_->visionPrefetchDispatches;'
    source=replace(source,old,old+'''
        auto vector=impl_->kernels->visionVector;
        // Same ordered FMA chain; only adjacent independent outputs share loads.
        // Original scalar route remains available for unsupported views/layouts.
        bool aligned=inputs.size()==3&&visionFields[3]&&visionFields[10]&&
            visionFields[9]%visionFields[10]==0&&(visionFields[9]/visionFields[10])%4==0&&
            visionFields[3]%4==0&&visionFields[5]%4==0&&visionFields[7]%4==0&&
            threads%visionFields[3]==0&&threads/visionFields[3]==visionFields[2]&&threads%4==0&&
            c.dataType==MPSDataTypeFloat32&&c.offset==0;
        for(const auto& input:inputs)aligned=aligned&&input.impl_->matrix.dataType==MPSDataTypeFloat32&&input.impl_->matrix.offset==0;
        if(aligned&&vector&&vector.threadExecutionWidth==32&&vector.maxTotalThreadsPerThreadgroup>=128&&
            vector.staticThreadgroupMemoryLength<=impl_->device.maxThreadgroupMemoryLength){
            pipeline=vector;vectorVisionContext=true;++impl_->visionVectorDispatches;
        }''')
    old='''if(serialVisionPrefetch) {
        [encoder dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(threads,size_t(128)),1,1)];'''
    source=replace(source,old,'''if(serialVisionPrefetch) {
        const size_t count=vectorVisionContext?threads/4:threads;
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(count,size_t(128)),1,1)];''')
    return source

def header(source):
    old='size_t LastVisionPrefetchDispatches() const;'
    return replace(source,old,old+'\n    size_t LastVisionVectorDispatches() const;')

def graph_header(source):
    source=replace(source,'size_t visionPrefetchDispatches_=0,mlpGraphDispatches_=0;','size_t visionPrefetchDispatches_=0,visionVectorDispatches_=0,mlpGraphDispatches_=0;')
    old='size_t LastVisionPrefetchDispatches() const {return visionPrefetchDispatches_;}'
    source=replace(source,old,old+'\n    size_t LastVisionVectorDispatches() const {return visionVectorDispatches_;}')
    old='APISET().AddProp("vision_prefetch_dispatches",&MetalCompiledGraph::LastVisionPrefetchDispatches);'
    return replace(source,old,old+'\n        APISET().AddProp("vision_vector_dispatches",&MetalCompiledGraph::LastVisionVectorDispatches);')

def graph(source):
    source=replace(source,'visionPrefetchDispatches_=0;mlpGraphDispatches_=0;','visionPrefetchDispatches_=0;visionVectorDispatches_=0;mlpGraphDispatches_=0;',2)
    old='visionPrefetchDispatches_=gpu_->LastVisionPrefetchDispatches();'
    return replace(source,old,old+'\n        visionVectorDispatches_=gpu_->LastVisionVectorDispatches();')

def entry(source):
    return replace(source,'info+="; serial_vision_prefetch";','info+="; serial_vision_prefetch; vector_vision_context4";')

def frame(source):
    old="serial_vision_prefetch_enabled='serial_vision_prefetch' in garnet.backend_info(),"
    return replace(source,old,old+"\n            vector_vision_context_enabled='vector_vision_context4' in garnet.backend_info(),\n            vision_vector_dispatches=vision.graph.vision_vector_dispatches if 'vector_vision_context4' in garnet.backend_info() else 0,")

def dense(source):
    old="vision_global_alpha_enabled='vision_global_alpha' in garnet.backend_info(),vision_global_alpha_dispatches=0,"
    return replace(source,old,old+"\n            vector_vision_context_enabled='vector_vision_context4' in garnet.backend_info(),vision_vector_dispatches=0,")

OPERATIONS={'src/backends/metal/metal_context.mm':context,'src/backends/metal/metal_context.h':header,
 'src/backends/metal/compiled_graph.cpp':graph,'src/backends/metal/compiled_graph.h':graph_header,
 'src/entry/metal_main.cpp':entry,'test2026/metal/frame_worker_garnet.py':frame,'test2026/metal/run_matched_garnet.py':dense}
