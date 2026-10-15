// Inspect actual hardware sampling support; validate two dependent compute passes.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
int main(){@autoreleasepool{try{
    auto device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("No Metal device");
    NSMutableDictionary* result=[NSMutableDictionary dictionaryWithDictionary:@{
        @"scope":@"Counter capability and exact dependency check only; no inference speed claim",
        @"device":device.name,
        @"stage_boundary":@([device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]),
        @"dispatch_boundary":@([device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]),
        @"blit_boundary":@([device supportsCounterSampling:MTLCounterSamplingPointAtBlitBoundary])}];
    NSMutableArray* names=[NSMutableArray array];id<MTLCounterSet> timestamps=nil;
    for(id<MTLCounterSet> set in device.counterSets){[names addObject:@{@"name":set.name,@"counters":@(set.counters.count)}];if([set.name isEqualToString:MTLCommonCounterSetTimestamp])timestamps=set;}
    result[@"counter_sets"]=names;
    if(timestamps&&[device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]){
        NSError* error=nil;auto descriptor=[MTLCounterSampleBufferDescriptor new];descriptor.counterSet=timestamps;descriptor.sampleCount=4;descriptor.storageMode=MTLStorageModeShared;
        auto counters=[device newCounterSampleBufferWithDescriptor:descriptor error:&error];if(!counters)throw std::runtime_error(error.localizedDescription.UTF8String);
        NSString* source=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void transform(device const float* a [[buffer(0)]],device float* b [[buffer(1)]],uint id [[thread_position_in_grid]]){if(id<4096)b[id]=a[id]*1.5f+0.25f;}";
        auto options=[MTLCompileOptions new];options.fastMathEnabled=NO;
        auto library=[device newLibraryWithSource:source options:options error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"transform"] error:&error];if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
        std::vector<float> values(4096),expected(4096);for(size_t i=0;i<values.size();++i){values[i]=float(i%17)*0.125f;expected[i]=(values[i]*1.5f+0.25f)*1.5f+0.25f;}
        auto input=[device newBufferWithBytes:values.data() length:values.size()*4 options:MTLResourceStorageModeShared];
        auto intermediate=[device newBufferWithLength:values.size()*4 options:MTLResourceStorageModeShared];auto output=[device newBufferWithLength:values.size()*4 options:MTLResourceStorageModeShared];
        auto queue=[device newCommandQueue];auto command=[queue commandBuffer];
        for(NSUInteger pass=0;pass<2;++pass){
            auto spec=[MTLComputePassDescriptor computePassDescriptor];auto attachment=spec.sampleBufferAttachments[0];attachment.sampleBuffer=counters;attachment.startOfEncoderSampleIndex=pass*2;attachment.endOfEncoderSampleIndex=pass*2+1;
            auto encoder=[command computeCommandEncoderWithDescriptor:spec];if(!encoder)throw std::runtime_error("Counter compute encoder unavailable");
            [encoder setComputePipelineState:pipeline];[encoder setBuffer:pass?intermediate:input offset:0 atIndex:0];[encoder setBuffer:pass?output:intermediate offset:0 atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder endEncoding];
        }
        [command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
        if(std::memcmp(output.contents,expected.data(),expected.size()*4))throw std::runtime_error("Counter passes changed exact dependency result");
        auto raw=[counters resolveCounterRange:NSMakeRange(0,4)];if(raw.length!=4*sizeof(MTLCounterResultTimestamp))throw std::runtime_error("Unexpected resolved timestamp size");
        auto samples=static_cast<const MTLCounterResultTimestamp*>(raw.bytes);NSMutableArray* times=[NSMutableArray array];
        for(size_t i=0;i<4;++i){if(!samples[i].timestamp)throw std::runtime_error("Zero timestamp");[times addObject:@(samples[i].timestamp)];}
        if(samples[1].timestamp<samples[0].timestamp||samples[3].timestamp<samples[2].timestamp)throw std::runtime_error("Nonmonotonic within-pass timestamps");
        result[@"exact_dependency_values"]=@(values.size());result[@"raw_timestamps"]=times;result[@"sampled_compute_passes"]=@2;
        result[@"command_gpu_span_ms"]=@((command.GPUEndTime-command.GPUStartTime)*1000);
    }else{result[@"sampled_compute_passes"]=@0;}
    auto json=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil];std::fwrite(json.bytes,1,json.length,stdout);std::puts("");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"counter_capability_failed: %s\n",e.what());return 1;}}}
