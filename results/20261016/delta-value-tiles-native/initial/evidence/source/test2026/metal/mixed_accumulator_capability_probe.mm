// Actual mixed-input/FP32-accumulator capability check; not a model speed test.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
int main(){@autoreleasepool{try{
    auto d=MTLCreateSystemDefaultDevice();NSError* error=nil;
    NSString* code=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void mixed(device const half* input [[buffer(0)]],device float* output [[buffer(1)]],device half* halfOutput [[buffer(2)]],constant float& initial [[buffer(3)]]){simdgroup_half8x8 a,b;simdgroup_load(a,input,8);simdgroup_load(b,input+64,8);simdgroup_float8x8 c=make_filled_simdgroup_matrix<float,8>(initial);simdgroup_half8x8 h=make_filled_simdgroup_matrix<half,8>(half(initial));for(uint i=0;i<128;++i){simdgroup_multiply_accumulate(c,a,b,c);simdgroup_multiply_accumulate(h,a,b,h);}simdgroup_store(c,output,8);simdgroup_store(h,halfOutput,8);}";
    auto options=[MTLCompileOptions new];options.fastMathEnabled=NO;
    auto library=[d newLibraryWithSource:code options:options error:&error];
    if(!library){NSDictionary* report=@{@"supported":@NO,@"device":d.name,@"phase":@"Metal shader compilation",@"diagnostic":error.localizedDescription,@"scope":@"Capability only; no model or speed claim"};auto bytes=[NSJSONSerialization dataWithJSONObject:report options:0 error:nil];std::fwrite(bytes.bytes,1,bytes.length,stdout);std::puts("");return 0;}
    auto pipeline=[d newComputePipelineStateWithFunction:[library newFunctionWithName:@"mixed"] error:&error];if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
    if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<32)throw std::runtime_error("Unsupported SIMD geometry");
    auto queue=[d newCommandQueue];NSMutableArray* cases=[NSMutableArray array];
    for(unsigned fixture=0;fixture<2;++fixture){
        const float value=fixture?std::ldexp(1.0f,-14):0.25f,initial=fixture?0.0f:2048.0f;
        const float expected=float(double(initial)+1024.0*double(value)*double(value));
        std::vector<__fp16> input(128,__fp16(value));auto a=[d newBufferWithBytes:input.data() length:input.size()*2 options:MTLResourceStorageModeShared];
        auto out=[d newBufferWithLength:64*4 options:MTLResourceStorageModeShared];auto halfOut=[d newBufferWithLength:64*2 options:MTLResourceStorageModeShared];
        auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline];[e setBuffer:a offset:0 atIndex:0];[e setBuffer:out offset:0 atIndex:1];[e setBuffer:halfOut offset:0 atIndex:2];[e setBytes:&initial length:sizeof(initial) atIndex:3];[e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Mixed matrix command failed");
        auto actual=static_cast<const float*>(out.contents);auto halfActual=static_cast<const __fp16*>(halfOut.contents);
        for(size_t i=0;i<64;++i)if(actual[i]!=expected||!std::isfinite(float(halfActual[i])))throw std::runtime_error("Mixed accumulator did not preserve FP32 fixture");
        [cases addObject:@{@"initial":@(initial),@"half_input":@(value),@"fp32_expected":@(expected),@"fp32_actual":@(actual[0]),@"half_accumulator_actual":@(float(halfActual[0])),@"exact_outputs":@64}];
    }
    auto bytes=[NSJSONSerialization dataWithJSONObject:@{@"supported":@YES,@"device":d.name,@"fp32_accumulator_fixtures":cases,@"exact_outputs":@128,@"scope":@"Precision capability only; not model admission or performance"} options:NSJSONWritingPrettyPrinted error:nil];std::fwrite(bytes.bytes,1,bytes.length,stdout);std::puts("");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"mixed_accumulator_failed: %s\n",e.what());return 1;}}}
