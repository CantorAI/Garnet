// Experimental extended-scale exact BF16 codec. Does not modify serving admission.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static uint16_t encode(uint16_t value,int scale) {
    uint16_t sign=value&0x8000,mag=value&0x7fff;
    if(!mag)return sign;
    int source=(mag>>7)&255,exponent=source-127+scale;
    if(!source||source==255||exponent>15||exponent<-24)throw std::runtime_error("Unsupported exact encoding");
    if(exponent>=-14)return sign|uint16_t((exponent+15)<<10)|uint16_t((mag&127)<<3);
    uint32_t significand=128+(mag&127);int shift=exponent+17;
    if(shift<0&&(significand&((1u<<(-shift))-1)))throw std::runtime_error("Encoding would discard bits");
    return sign|uint16_t(shift>=0?significand<<shift:significand>>(-shift));
}

int main() {@autoreleasepool {try {
    auto device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("No Metal device");
    auto queue=[device newCommandQueue];
    std::vector<uint16_t> encoded;std::vector<int32_t> scales;std::vector<uint32_t> expected;
    for(uint32_t value=0;value<65536;++value) {
        auto magnitude=value&0x7fff;int exponent=(magnitude>>7)&255;
        if(magnitude&&(exponent==0||exponent==255))continue;
        int scale=magnitude?142-exponent:0;
        encoded.push_back(encode(value,scale));scales.push_back(scale);expected.push_back(value<<16);
    }
    auto halfBuffer=[device newBufferWithBytes:encoded.data() length:encoded.size()*2 options:MTLResourceStorageModeShared];
    auto scaleBuffer=[device newBufferWithBytes:scales.data() length:scales.size()*4 options:MTLResourceStorageModeShared];
    auto out=[device newBufferWithLength:expected.size()*4 options:MTLResourceStorageModeShared];
    if(!halfBuffer||!scaleBuffer||!out)throw std::runtime_error("Allocation failed");
    NSString* source=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void restore(device const half* h [[buffer(0)]],device const int* s [[buffer(1)]],device uint* o [[buffer(2)]],constant uint& count [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i<count)o[i]=as_type<uint>(ldexp(float(h[i]),-s[i]));}";
    bool anyFailure=false;
    for(bool fast:{false,true}) {
        MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=fast;NSError* error=nil;
        auto library=[device newLibraryWithSource:source options:options error:&error];
        if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"restore"] error:&error];
        if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];[encoder setBuffer:halfBuffer offset:0 atIndex:0];
        [encoder setBuffer:scaleBuffer offset:0 atIndex:1];[encoder setBuffer:out offset:0 atIndex:2];
        uint32_t count=expected.size();[encoder setBytes:&count length:4 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
        [command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
        auto actual=static_cast<const uint32_t*>(out.contents);size_t mismatches=0,zerosLost=0;int minFail=999,maxFail=-999;
        for(size_t i=0;i<expected.size();++i)if(actual[i]!=expected[i]) {
            ++mismatches;minFail=std::min(minFail,int(scales[i]));maxFail=std::max(maxFail,int(scales[i]));
            if(!(expected[i]&0x7fffffff))++zerosLost;
            if(mismatches<=8)printf("{\"kind\":\"mismatch\",\"fast_math\":%s,\"source_bits\":%u,\"actual_bits\":%u,\"scale\":%d}\n",fast?"true":"false",expected[i],actual[i],scales[i]);
        }
        printf("{\"kind\":\"summary\",\"fast_math\":%s,\"tested\":%zu,\"bit_mismatches\":%zu,\"signed_zero_mismatches\":%zu,\"failed_scale_min\":%d,\"failed_scale_max\":%d}\n",fast?"true":"false",expected.size(),mismatches,zerosLost,mismatches?minFail:0,mismatches?maxFail:0);
        anyFailure|=mismatches!=0;fflush(stdout);
    }
    return anyFailure?2:0;
}catch(const std::exception& error){fprintf(stderr,"extended_scale_failed: %s\n",error.what());return 1;}}}
