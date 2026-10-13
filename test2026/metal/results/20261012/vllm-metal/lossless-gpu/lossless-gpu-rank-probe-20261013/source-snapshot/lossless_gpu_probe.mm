// Isolated Metal microbenchmark. This is not full model inference.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

struct Params {uint64_t rows,inner,columns,aStride,oStride;};
static id<MTLBuffer> load(id<MTLDevice> device,NSString* folder,NSString* name) {
    NSData* bytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:name]];
    if(!bytes)throw std::runtime_error("Missing input bytes");
    if(bytes.length==0)return [device newBufferWithLength:1 options:MTLResourceStorageModeShared];
    return [device newBufferWithBytes:bytes.bytes length:bytes.length options:MTLResourceStorageModeShared];
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device,id<MTLLibrary> lib,NSString* name) {
    NSError* error=nil;
    auto p=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&error];
    if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);
    if(p.threadExecutionWidth!=32||p.maxTotalThreadsPerThreadgroup<128)throw std::runtime_error("Unsupported SIMD width");
    return p;
}
static void finish(id<MTLCommandBuffer> command) {
    [command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
}
int main(int argc,char** argv) {
    @autoreleasepool {try {
        if(argc!=3)throw std::runtime_error("Expected shader file and asset folder");
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if(!device)throw std::runtime_error("No Metal device");
        NSError* error=nil;
        NSString* shader=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
        auto lib=[device newLibraryWithSource:shader options:nil error:&error];
        if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto rawPipeline=pipeline(device,lib,@"bf16_decode"),packedPipeline=pipeline(device,lib,@"packed_decode");
        auto unpackPipeline=pipeline(device,lib,@"unpack_bits");auto queue=[device newCommandQueue];
        NSString* root=[NSString stringWithUTF8String:argv[2]];
        NSArray* children=[[[NSFileManager defaultManager] contentsOfDirectoryAtPath:root error:&error] sortedArrayUsingSelector:@selector(compare:)];
        for(NSString* child in children) {
            NSString* folder=[root stringByAppendingPathComponent:child];
            NSData* caseBytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"case.json"]];
            if(!caseBytes)continue;
            NSDictionary* info=[NSJSONSerialization JSONObjectWithData:caseBytes options:0 error:&error];
            uint64_t count=[info[@"count"] unsignedLongLongValue],blocks=(count+127)/128;
            auto raw=load(device,folder,@"raw.bin"),payload=load(device,folder,@"payload.bin");
            auto tags=load(device,folder,@"packed.bin"),metadata=load(device,folder,@"metadata.bin"),exceptions=load(device,folder,@"exceptions.bin");
            auto ranks=load(device,folder,@"ranks.bin");
            auto unpacked=[device newBufferWithLength:blocks*128*2 options:MTLResourceStorageModeShared];
            auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
            [encoder setComputePipelineState:unpackPipeline];
            [encoder setBuffer:payload offset:0 atIndex:0];[encoder setBuffer:tags offset:0 atIndex:1];
            [encoder setBuffer:metadata offset:0 atIndex:2];[encoder setBuffer:exceptions offset:0 atIndex:3];
            [encoder setBuffer:unpacked offset:0 atIndex:4];[encoder setBytes:&blocks length:8 atIndex:5];
            [encoder setBuffer:ranks offset:0 atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake((blocks+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [encoder endEncoding];finish(command);
            if(std::memcmp(raw.contents,unpacked.contents,count*2))throw std::runtime_error("GPU bit roundtrip mismatch");
            std::printf("{\"kind\":\"roundtrip\",\"case\":\"%s\",\"weights\":%llu,\"gpu_bits_exact\":true}\n",child.UTF8String,(unsigned long long)count);std::fflush(stdout);
            if(![info[@"finite"] boolValue])continue;
            uint64_t columns=[info[@"rows"] unsignedLongLongValue],inner=[info[@"inner"] unsignedLongLongValue];
            if(inner%128)throw std::runtime_error("Probe needs aligned inner dimension");
            std::vector<float> input(inner*2);
            for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*0.037f)*0.3f;
            auto a=[device newBufferWithBytes:input.data() length:input.size()*4 options:MTLResourceStorageModeShared];
            auto output=[device newBufferWithLength:columns*2*4 options:MTLResourceStorageModeShared];
            for(uint64_t batch=1;batch<=2;++batch) {
                Params p{batch,inner,columns,inner,columns};std::vector<float> reference(columns*batch);
                auto run=[&](bool packed) {
                    auto begin=std::chrono::steady_clock::now();auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];
                    [enc setComputePipelineState:packed?packedPipeline:rawPipeline];
                    [enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:packed?payload:raw offset:0 atIndex:1];
                    [enc setBuffer:output offset:0 atIndex:2];[enc setBytes:&p length:sizeof(p) atIndex:3];
                    if(packed){[enc setBuffer:tags offset:0 atIndex:4];[enc setBuffer:metadata offset:0 atIndex:5];[enc setBuffer:exceptions offset:0 atIndex:6];[enc setBuffer:ranks offset:0 atIndex:7];}
                    [enc dispatchThreadgroups:MTLSizeMake((columns+15)/16,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                    [enc endEncoding];finish(cb);
                    return std::pair<double,double>{(cb.GPUEndTime-cb.GPUStartTime)*1000,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()};
                };
                run(false);std::memcpy(reference.data(),output.contents,reference.size()*4);run(true);
                float maximumError=0,maximumValue=0;auto actual=static_cast<float*>(output.contents);
                for(size_t i=0;i<reference.size();++i){if(!std::isfinite(actual[i])||!std::isfinite(reference[i]))throw std::runtime_error("Nonfinite projection");maximumError=std::max(maximumError,std::abs(reference[i]-actual[i]));maximumValue=std::max(maximumValue,std::abs(reference[i]));}
                bool exact=std::memcmp(reference.data(),actual,reference.size()*4)==0;
                if(maximumError>1e-5f*(1+maximumValue))throw std::runtime_error("Projection numerical mismatch");
                std::printf("{\"kind\":\"projection_check\",\"case\":\"%s\",\"batch\":%llu,\"output_bits_exact\":%s,\"max_abs_error\":%.9g}\n",child.UTF8String,(unsigned long long)batch,exact?"true":"false",maximumError);
                // ABBA rounds, each policy completes before the next begins.
                for(int round=0;round<4;++round)for(bool packed:{false,true,true,false}) {
                    for(int warm=0;warm<3;++warm)run(packed);
                    std::vector<double> gpu,wall;
                    for(int trial=0;trial<12;++trial){auto timing=run(packed);gpu.push_back(timing.first);wall.push_back(timing.second);}
                    std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());
                    std::printf("{\"kind\":\"timing\",\"case\":\"%s\",\"batch\":%llu,\"round\":%d,\"policy\":\"%s\",\"samples\":12,\"gpu_ms_median\":%.9g,\"wall_ms_median\":%.9g}\n",child.UTF8String,(unsigned long long)batch,round,packed?"packed":"raw",(gpu[5]+gpu[6])*0.5,(wall[5]+wall[6])*0.5);std::fflush(stdout);
                }
            }
        }
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"lossless_probe_failed: %s\n",error.what());return 1;}}
}
