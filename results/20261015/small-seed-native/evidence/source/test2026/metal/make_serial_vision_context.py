"""Original vision softmax/context with threadgroup or ordered-prefetch variants."""
def generate(native,harness):
    start=native.index('struct VisionParams {');params=native[start:native.index('};',start)+2]
    start=native.index('kernel void vision_softmax(');softmax=native[start:native.index('// Input/output matrices',start)]
    start=native.index('kernel void vision_op(');context=native[start:native.index(')METAL"]',start)]
    shader='#include <metal_stdlib>\nusing namespace metal;\n'+params+'\n'+softmax+context
    for width in (4,8):
        code='''
kernel void vision_prefetchWIDTH(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    ulong row=id/p.outColumns,col=id%p.outColumns;if(row>=p.outRows)return;
    ulong dim=p.hidden/p.heads,head=col/dim,d=col%dim,batchBase=(row/p.tokens)*p.tokens;
    float result=0;ulong key=0;
    for(;key+WIDTH<=p.tokens;key+=WIDTH) {
LOADS
ADDS
    }
    for(;key<p.tokens;++key)result=fma(a[row*p.aStride+head*p.tokens+key],b[(batchBase+key)*p.bStride+2*p.hidden+head*dim+d],result);
    out[row*p.oStride+col]=result;
}
'''.replace('WIDTH',str(width))
        loads='\n'.join('        float p%d=a[row*p.aStride+head*p.tokens+key+%d],v%d=b[(batchBase+key+%d)*p.bStride+2*p.hidden+head*dim+d];'%(i,i,i,i) for i in range(width))
        adds='\n'.join('        result=fma(p%d,v%d,result);'%(i,i) for i in range(width))
        shader+=code.replace('LOADS',loads).replace('ADDS',adds)
    cpp=harness.replace('if(argc!=2)','if(argc!=3)').replace('Expected extracted shader','Expected extracted shader and variant').replace('#include <vector>','#include <vector>\n#include <string>')
    needle='    id<MTLComputePipelineState> kernels[3];';assert cpp.count(needle)==1
    start=cpp.index(needle);end=cpp.index('    for(uint64_t batch:',start)
    cpp=cpp[:start]+'''    auto strictOptions=[MTLCompileOptions new];strictOptions.fastMathEnabled=NO;
    auto strictLibrary=[d newLibraryWithSource:source options:strictOptions error:&error];
    if(!strictLibrary)throw std::runtime_error(error.localizedDescription.UTF8String);
    id<MTLComputePipelineState> kernels[4];NSArray* names=@[@"vision_softmax",@"vision_op",@"vision_prefetch4",@"vision_prefetch8"];
    for(int i=0;i<4;++i){auto selected=i<2?library:strictLibrary;kernels[i]=[d newComputePipelineStateWithFunction:[selected newFunctionWithName:names[i]] error:&error];if(!kernels[i])throw std::runtime_error("Pipeline failed");}
    const int variant=std::stoi(argv[2]);if(variant<0||variant>4)throw std::runtime_error("Unknown variant");
    const uint64_t candidateTG[]={64,128,256,128,128};
    auto candidatePipeline=variant<3?kernels[1]:kernels[variant-1];
    if(candidatePipeline.threadExecutionWidth!=32||candidatePipeline.maxTotalThreadsPerThreadgroup<candidateTG[variant])throw std::runtime_error("Candidate device limit");
'''+cpp[end:]
    start=cpp.index('        auto encode=');end=cpp.index('        auto run=',start)
    cpp=cpp[:start]+'''        auto encode=[&](id<MTLCommandBuffer> cb,int mode) {
            bool contraction=mode!=0;auto pipeline=mode==0?kernels[0]:mode==1?kernels[1]:candidatePipeline;
            auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:pipeline];
            Params p{rows,contraction?heads*tokens:hidden*3,rows,contraction?hidden:heads*tokens,contraction?prob.stride:qkv.stride,contraction?qkv.stride:scores.stride,1,contraction?out.stride:prob.stride,tokens,hidden,heads,4,contraction?8ul:8ul};
            [enc setBuffer:contraction?prob.data:qkv.data offset:0 atIndex:0];[enc setBuffer:contraction?qkv.data:scores.data offset:0 atIndex:1];
            [enc setBuffer:qkv.data offset:0 atIndex:2];[enc setBuffer:contraction?out.data:prob.data offset:0 atIndex:3];[enc setBytes:&p length:sizeof(p) atIndex:4];
            if(contraction){uint64_t limit=mode==2?candidateTG[variant]:kernels[1].maxTotalThreadsPerThreadgroup;[enc dispatchThreads:MTLSizeMake(rows*hidden,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(limit,rows*hidden),1,1)];}
            else {[enc setThreadgroupMemoryLength:(tokens*4+15)&~uint64_t(15) atIndex:0];[enc dispatchThreadgroups:MTLSizeMake(rows*heads,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];}
            [enc endEncoding];
        };
'''+cpp[end:]
    old='if(fused)encode(cb,2);else {encode(cb,0);encode(cb,1);}';assert cpp.count(old)==1
    cpp=cpp.replace(old,'encode(cb,0);encode(cb,fused?2:1);')
    cpp=cpp.replace('        if(!exact||worst>1)throw std::runtime_error("Correctness failed");','        bool admitted=exact&&worst<=1;')
    cpp=cpp.replace('\\"bits_exact\\":true,','\\"variant\\":%d,\\"passed\\":%s,\\"bits_exact\\":%s,')
    cpp=cpp.replace('batch,tokens,heads,dim,worst,peak);','batch,tokens,heads,dim,variant,admitted?"true":"false",exact?"true":"false",worst,peak);')
    needle='        for(int round=0;round<2;++round)';assert cpp.count(needle)==1
    cpp=cpp.replace(needle,'''        if(!admitted)continue;
        auto commonWarm=std::chrono::steady_clock::now();size_t warmEpoch=0;
        do{run((warmEpoch++%2)!=0);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-commonWarm).count()<.25);
'''+needle)
    needle='            std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());';assert cpp.count(needle)==1
    cpp=cpp.replace(needle,'''            auto rawGpu=gpu,rawWall=wall;
            printf("{\\"kind\\":\\"samples\\",\\"batch\\":%llu,\\"tokens\\":%llu,\\"heads\\":%llu,\\"dim\\":%llu,\\"variant\\":%d,\\"round\\":%d,\\"candidate\\":%s,\\"gpu_ms\\":[",batch,tokens,heads,dim,variant,round,fused?"true":"false");
            for(size_t i=0;i<rawGpu.size();++i)printf("%s%.12g",i?",":"",rawGpu[i]);printf("],\\"wall_ms\\":[");
            for(size_t i=0;i<rawWall.size();++i)printf("%s%.12g",i?",":"",rawWall[i]);puts("]}");
'''+needle)
    return shader,cpp
