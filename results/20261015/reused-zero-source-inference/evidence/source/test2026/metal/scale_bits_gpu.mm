// Experimental GPU bit scaling checked against host IEEE ldexpf.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
int main(int argc,char** argv) {@autoreleasepool {try {
    if(argc!=2)throw std::runtime_error("Expected experimental Metal source");
    auto device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("No Metal device");auto queue=[device newCommandQueue];
    std::vector<uint32_t> input,expected;std::vector<int32_t> scales;
    auto append=[&](uint32_t bits,int scale){float x;std::memcpy(&x,&bits,4);float result=std::ldexp(x,scale);uint32_t out;std::memcpy(&out,&result,4);input.push_back(bits);scales.push_back(scale);expected.push_back(out);};
    for(uint32_t bf16=0;bf16<65536;++bf16) {
        if(((bf16>>7)&255)==255)continue;
        for(int scale:{-141,-138,-112,-64,-24,-1,0,1,24,64,112,138,141})append(bf16<<16,scale);
    }
    for(uint32_t sign:{0u,0x80000000u})for(uint32_t exponent=0;exponent<255;++exponent)
        for(uint32_t fraction:{0u,1u,127u,0x3fffffu,0x7ffffeu,0x7fffffu})
            for(int scale:{-141,-138,-112,-64,-24,-1,0,1,24,64,112,138,141})append(sign|(exponent<<23)|fraction,scale);
    auto in=[device newBufferWithBytes:input.data() length:input.size()*4 options:MTLResourceStorageModeShared];
    auto scaleBuffer=[device newBufferWithBytes:scales.data() length:scales.size()*4 options:MTLResourceStorageModeShared];
    auto out=[device newBufferWithLength:expected.size()*4 options:MTLResourceStorageModeShared];
    if(!in||!scaleBuffer||!out)throw std::runtime_error("Allocation failed");NSError* error=nil;
    NSString* helper=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
    if(!helper)throw std::runtime_error(error.localizedDescription.UTF8String);
    NSString* source=[@"#include <metal_stdlib>\nusing namespace metal;\n" stringByAppendingString:helper];
    source=[source stringByAppendingString:@"\nkernel void restore(device const uint* input [[buffer(0)]],device const int* scale [[buffer(1)]],device uint* output [[buffer(2)]],constant uint& count [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i<count)output[i]=scale_bits(input[i],scale[i]);}"];
    bool failed=false;
    for(bool fast:{false,true}) {
        MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=fast;
        auto library=[device newLibraryWithSource:source options:options error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"restore"] error:&error];if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];
        [encoder setBuffer:in offset:0 atIndex:0];[encoder setBuffer:scaleBuffer offset:0 atIndex:1];[encoder setBuffer:out offset:0 atIndex:2];
        uint32_t count=expected.size();[encoder setBytes:&count length:4 atIndex:3];[encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
        auto actual=static_cast<const uint32_t*>(out.contents);size_t mismatches=0;
        for(size_t i=0;i<expected.size();++i)if(actual[i]!=expected[i]){++mismatches;if(mismatches<=8)printf("{\"kind\":\"mismatch\",\"input_bits\":%u,\"scale\":%d,\"expected_bits\":%u,\"actual_bits\":%u}\n",input[i],scales[i],expected[i],actual[i]);}
        printf("{\"kind\":\"summary\",\"fast_math\":%s,\"tested\":%zu,\"bit_mismatches\":%zu}\n",fast?"true":"false",expected.size(),mismatches);fflush(stdout);failed|=mismatches!=0;
    }
    return failed?2:0;
}catch(const std::exception& e){fprintf(stderr,"scale_bits_failed: %s\n",e.what());return 1;}}}
