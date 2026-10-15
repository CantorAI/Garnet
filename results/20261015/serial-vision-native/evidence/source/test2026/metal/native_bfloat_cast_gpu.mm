// Exhaustive finite BF16 load-to-FP32 bit compatibility for the native type.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <stdexcept>
#include <vector>
int main() {@autoreleasepool {try {
    auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];
    std::vector<uint16_t> bits;for(uint32_t i=0;i<65536;++i)if(((i>>7)&255)!=255)bits.push_back(i);
    auto input=[device newBufferWithBytes:bits.data() length:bits.size()*2 options:MTLResourceStorageModeShared];
    auto output=[device newBufferWithLength:bits.size()*4 options:MTLResourceStorageModeShared];
    if(!device||!queue||!input||!output)throw std::runtime_error("Allocation/device unavailable");
    NSString* source=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void load_bits(device const bfloat* input [[buffer(0)]],device uint* output [[buffer(1)]],constant uint& count [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<count)output[i]=as_type<uint>(float(input[i]));}";
    bool failed=false;
    for(bool fast:{false,true}) {
        MTLCompileOptions* options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_1;options.fastMathEnabled=fast;NSError* error=nil;
        auto library=[device newLibraryWithSource:source options:options error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"load_bits"] error:&error];if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];[encoder setBuffer:input offset:0 atIndex:0];[encoder setBuffer:output offset:0 atIndex:1];
        uint32_t count=bits.size();[encoder setBytes:&count length:4 atIndex:2];[encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
        auto actual=static_cast<const uint32_t*>(output.contents);size_t mismatches=0;
        for(size_t i=0;i<bits.size();++i)if(actual[i]!=(uint32_t(bits[i])<<16)){++mismatches;if(mismatches<=8)printf("{\"kind\":\"mismatch\",\"source_bits\":%u,\"actual_bits\":%u}\n",bits[i],actual[i]);}
        printf("{\"kind\":\"summary\",\"tested\":%zu,\"fast_math\":%s,\"bit_mismatches\":%zu}\n",bits.size(),fast?"true":"false",mismatches);fflush(stdout);failed|=mismatches!=0;
    }
    return failed?2:0;
}catch(const std::exception& e){fprintf(stderr,"native_bfloat_cast_failed: %s\n",e.what());return 1;}}}
