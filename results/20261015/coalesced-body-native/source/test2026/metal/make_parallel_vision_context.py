"""Split the FP32 value contraction across four SIMD groups; softmax is unchanged."""
from pathlib import Path

def generate(native,harness):
    begin=native.index('struct VisionParams {');params=native[begin:native.index('};',begin)+2]
    begin=native.index('kernel void vision_softmax_context(');end=native.index(')METAL',begin)
    control=native[begin:end]
    original='''    for(ulong d=tid;d<dim;d+=128) {
        float result=0;
        for(ulong key=0;key<p.tokens;++key)result+=scratch[key]*a[(batchBase+key)*p.aStride+2*p.hidden+head*dim+d];
        out[row*p.oStride+head*dim+d]=result;
    }'''
    assert control.count(original)==1
    candidate=control.replace('vision_softmax_context(', 'vision_parallel_context(',1).replace(original,'''    threadgroup float valuePartials[4*128];
    for(ulong d=lane;d<dim;d+=32) {
        float result=0;
        for(ulong key=simd;key<p.tokens;key+=4)result+=scratch[key]*a[(batchBase+key)*p.aStride+2*p.hidden+head*dim+d];
        valuePartials[simd*128+d]=result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(ulong d=tid;d<dim;d+=128) {
        float result=valuePartials[d]+valuePartials[128+d]+valuePartials[256+d]+valuePartials[384+d];
        out[row*p.oStride+head*dim+d]=result;
    }''')
    shader='#include <metal_stdlib>\nusing namespace metal;\n'+params+'\n'+control+candidate
    cpp=harness.replace('id<MTLComputePipelineState> kernels[3];','id<MTLComputePipelineState> kernels[2];').replace('NSArray* names=@[@"vision_softmax",@"vision_op",@"vision_softmax_context"];','NSArray* names=@[@"vision_softmax_context",@"vision_parallel_context"];').replace('i<3;++i','i<2;++i')
    start=cpp.index('            auto enc=[cb computeCommandEncoder];');stop=cpp.index('            [enc endEncoding];',start)
    cpp=cpp[:start]+'''            auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:kernels[mode]];
            Params p{rows,hidden*3,rows,hidden,qkv.stride,scores.stride,1,out.stride,tokens,hidden,heads,4,10};
            [enc setBuffer:qkv.data offset:0 atIndex:0];[enc setBuffer:scores.data offset:0 atIndex:1];
            [enc setBuffer:qkv.data offset:0 atIndex:2];[enc setBuffer:out.data offset:0 atIndex:3];
            [enc setBytes:&p length:sizeof(p) atIndex:4];
            [enc setThreadgroupMemoryLength:(tokens*4+15)&~uint64_t(15) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(rows*heads,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
'''+cpp[stop:]
    cpp=cpp.replace('if(fused)encode(cb,2);else {encode(cb,0);encode(cb,1);}','encode(cb,fused?1:0);')
    cpp=cpp.replace('bool exact=true;double peak=0,worst=0;','bool exact=true;double peak=0,worst=0,controlWorst=0;')
    needle='                    peak=std::max(peak,std::abs(value-double(baseline[index])));'
    assert cpp.count(needle)==1
    cpp=cpp.replace(needle,needle+'''
                    if(std::abs(value-double(baseline[index]))>5e-6+5e-5*std::abs(double(baseline[index])))throw std::runtime_error("Original candidate/control envelope exceeded");
                    controlWorst=std::max(controlWorst,std::abs(double(baseline[index])-expected)/(5e-6+5e-5*std::abs(expected)));''')
    cpp=cpp.replace('if(!exact||worst>1)','if(worst>1||controlWorst>1)')
    cpp=cpp.replace('\\"bits_exact\\":true,','\\"bits_exact\\":%s,\\"control_fp64_worst_ratio\\":%.9g,')
    cpp=cpp.replace('batch,tokens,heads,dim,worst,peak);','batch,tokens,heads,dim,exact?"true":"false",controlWorst,worst,peak);')
    needle='        for(int round=0;round<2;++round)'
    cpp=cpp.replace(needle,'''        auto commonWarm=std::chrono::steady_clock::now();size_t warmEpoch=0;
        do{run((warmEpoch++%2)!=0);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-commonWarm).count()<.25);
'''+needle)
    cpp=cpp.replace('std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());','''auto rawGpu=gpu,rawWall=wall;
            std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());
            printf("{\\"kind\\":\\"samples\\",\\"batch\\":%llu,\\"tokens\\":%llu,\\"heads\\":%llu,\\"dim\\":%llu,\\"round\\":%d,\\"candidate\\":%s,\\"gpu_ms\\":[",batch,tokens,heads,dim,round,fused?"true":"false");
            for(size_t i=0;i<rawGpu.size();++i)printf("%s%.12g",i?",":"",rawGpu[i]);printf("],\\"wall_ms\\":[");
            for(size_t i=0;i<rawWall.size();++i)printf("%s%.12g",i?",":"",rawWall[i]);puts("]}");''')
    return shader,cpp

if __name__=='__main__':
    import sys
    folder=Path(__file__).resolve().parent
    shader,cpp=generate((folder.parents[1]/'src/backends/metal/metal_context.mm').read_text(),(folder/'vision_fused_context_gpu.mm').read_text())
    Path(sys.argv[1]).write_text(shader);Path(sys.argv[2]).write_text(cpp)
