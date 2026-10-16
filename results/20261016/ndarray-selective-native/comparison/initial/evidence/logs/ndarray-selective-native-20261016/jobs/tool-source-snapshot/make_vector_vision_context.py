"""Vectorize adjacent dimensions; preserve original per-output FMA order and scalar control."""
from make_serial_vision_context import generate as serial_generate

def generate(native,harness,selected_header):
    shader,cpp=serial_generate(native,harness)
    shader=shader[:shader.index('kernel void vision_prefetch4(')]
    control=selected_header.split('R"GARNET(',1)[1].split(')GARNET"',1)[0]
    control=control[control.index('kernel void vision_prefetch8('):]
    assert control.count('result=fma(')==9
    shader+=control
    for width in (2,4):
        code='''
kernel void vision_vectorWIDTH(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    ulong columns=p.outColumns/WIDTH,row=id/columns,col=(id%columns)*WIDTH;if(row>=p.outRows)return;
    ulong dim=p.hidden/p.heads,head=col/dim,d=col%dim,batchBase=(row/p.tokens)*p.tokens;
    floatWIDTH result=0;ulong key=0;
    for(;key+8<=p.tokens;key+=8) {
LOADS
ADDS
    }
    for(;key<p.tokens;++key)result=fma(floatWIDTH(a[row*p.aStride+head*p.tokens+key]),*(device const floatWIDTH*)(b+(batchBase+key)*p.bStride+2*p.hidden+head*dim+d),result);
    *(device floatWIDTH*)(out+row*p.oStride+col)=result;
}
'''.replace('WIDTH',str(width))
        loads='\n'.join(f'        float p{i}=a[row*p.aStride+head*p.tokens+key+{i}];float{width} v{i}=*(device const float{width}*)(b+(batchBase+key+{i})*p.bStride+2*p.hidden+head*dim+d);' for i in range(8))
        adds='\n'.join(f'        result=fma(float{width}(p{i}),v{i},result);' for i in range(8))
        shader+=code.replace('LOADS',loads).replace('ADDS',adds)
    start=cpp.index('    id<MTLComputePipelineState> kernels[4];');end=cpp.index('    for(uint64_t batch:',start)
    cpp=cpp[:start]+'''    id<MTLComputePipelineState> kernels[4];NSArray* names=@[@"vision_softmax",@"vision_prefetch8",@"vision_vector2",@"vision_vector4"];
    for(int i=0;i<4;++i){auto selected=i==0?library:strictLibrary;kernels[i]=[d newComputePipelineStateWithFunction:[selected newFunctionWithName:names[i]] error:&error];if(!kernels[i])throw std::runtime_error("Pipeline failed");}
    const int variant=std::stoi(argv[2]);if(variant<0||variant>2)throw std::runtime_error("Unknown variant");
    const uint64_t candidateTG[]={128,128,128},candidateWidth[]={1,2,4};
    auto candidatePipeline=variant==0?kernels[1]:kernels[variant+1];
    for(auto pipeline:{kernels[1],candidatePipeline})if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128)throw std::runtime_error("Candidate device limit");
'''+cpp[end:]
    needle='        auto encode=[&](id<MTLCommandBuffer> cb,int mode)';assert cpp.count(needle)==1
    cpp=cpp.replace(needle,'''        if(hidden%candidateWidth[variant]||dim%candidateWidth[variant]||qkv.stride%candidateWidth[variant]||out.stride%candidateWidth[variant])throw std::runtime_error("Vector alignment precondition");
'''+needle)
    old='if(contraction){uint64_t limit=mode==2?candidateTG[variant]:kernels[1].maxTotalThreadsPerThreadgroup;[enc dispatchThreads:MTLSizeMake(rows*hidden,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(limit,rows*hidden),1,1)];}'
    assert cpp.count(old)==1
    cpp=cpp.replace(old,'if(contraction){uint64_t count=rows*hidden/(mode==2?candidateWidth[variant]:1);[enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(uint64_t(128),count),1,1)];}')
    assert 'strictOptions.fastMathEnabled=NO' in cpp and 'if(!admitted)continue;' in cpp
    return shader,cpp
