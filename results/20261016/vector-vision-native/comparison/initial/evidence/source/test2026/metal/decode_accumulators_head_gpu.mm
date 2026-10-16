// FP32-accumulated BF16 decode independent-accumulator experiment, separate from serving inference.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
struct Params {uint64_t rows,inner,columns,aStride,oStride;};
struct Tile {const char* name;uint64_t rows,columns,inner,threads;id<MTLComputePipelineState> pipeline;};
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device,id<MTLLibrary> library,const char* name) {
    NSError* error=nil;auto result=[device newComputePipelineStateWithFunction:[library newFunctionWithName:[NSString stringWithUTF8String:name]] error:&error];
    if(!result)throw std::runtime_error(error.localizedDescription.UTF8String);return result;
}
static void finish(id<MTLCommandBuffer> command) {
    [command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
}
int main(int argc,char** argv) {
    @autoreleasepool {try {
        if(argc!=3)throw std::runtime_error("Expected shader and assets");
        auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError* error=nil;
        NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
        auto library=[device newLibraryWithSource:source options:nil error:&error];
        if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
        std::vector<Tile> tiles={{"bf16_decode",1,16,4,128,nil},{"bf16_decode_a2_t64",1,8,4,64,nil}};
        for(auto& tile:tiles){tile.pipeline=pipeline(device,library,tile.name);if(tile.pipeline.threadExecutionWidth!=32||tile.pipeline.maxTotalThreadsPerThreadgroup<tile.threads||tile.pipeline.staticThreadgroupMemoryLength>device.maxThreadgroupMemoryLength)throw std::runtime_error("Tile exceeds actual device limits");}
        NSString* folder=[NSString stringWithUTF8String:argv[2]];NSData* data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];
        NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        for(NSDictionary* item in manifest[@"cases"]) {@autoreleasepool {
            NSString* name=item[@"name"];uint64_t columns=[item[@"columns"] unsignedLongLongValue],inner=[item[@"inner"] unsignedLongLongValue];
            if(columns<100000)continue; // All manifest vocabulary heads; excludes body and synthetic cases.
            NSData* bytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:[name stringByAppendingString:@".bin"]]];
            if(bytes.length!=inner*columns*2)throw std::runtime_error("Incomplete BF16 matrix");
            auto weight=[device newBufferWithBytes:bytes.bytes length:bytes.length options:MTLResourceStorageModeShared];
            for(uint64_t rows:{1,2}) {
                uint64_t aStride=inner+8,oStride=columns+7;std::vector<float> input(rows*aStride,0);
                for(uint64_t row=0;row<rows;++row)for(uint64_t k=0;k<inner;++k)input[row*aStride+k]=std::sin(float(row*inner+k)*0.013f)*0.3f;
                auto a=[device newBufferWithBytes:input.data() length:input.size()*4 options:MTLResourceStorageModeShared];
                auto out=[device newBufferWithLength:rows*oStride*4 options:MTLResourceStorageModeShared];Params p{rows,inner,columns,aStride,oStride};
                auto run=[&](const Tile& tile) {
                    auto begin=std::chrono::steady_clock::now();auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];
                    [enc setComputePipelineState:tile.pipeline];[enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:weight offset:0 atIndex:1];[enc setBuffer:out offset:0 atIndex:2];[enc setBytes:&p length:sizeof(p) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake((columns+tile.columns-1)/tile.columns,1,1) threadsPerThreadgroup:MTLSizeMake(tile.threads,1,1)];[enc endEncoding];finish(cb);
                    return std::pair<double,double>{(cb.GPUEndTime-cb.GPUStartTime)*1000,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()};
                };
                std::fill_n(static_cast<float*>(out.contents),rows*oStride,99.5f);run(tiles[0]);std::vector<float> reference(rows*oStride);std::memcpy(reference.data(),out.contents,reference.size()*4);
                // Independent host FP64 dot products and a forward-error bound.
                // The bound is fixed before candidate execution; control must pass too.
                const auto* rawWeight=static_cast<const uint16_t*>(bytes.bytes);
                std::vector<double> oracle(rows*columns),bounds(rows*columns);
                double n=std::ceil(double(inner)/32.0)+32.0;
                double gamma=n*std::numeric_limits<float>::epsilon()/(1.0-n*std::numeric_limits<float>::epsilon());
                for(uint64_t row=0;row<rows;++row)for(uint64_t col=0;col<columns;++col) {
                    double sum=0,absoluteProducts=0;
                    for(uint64_t k=0;k<inner;++k) {
                        uint32_t bits=uint32_t(rawWeight[col*inner+k])<<16;float w;std::memcpy(&w,&bits,4);
                        double product=double(input[row*aStride+k])*double(w);sum+=product;absoluteProducts+=std::abs(product);
                    }
                    oracle[row*columns+col]=sum;bounds[row*columns+col]=gamma*absoluteProducts+1e-7;
                    if(std::abs(double(reference[row*oStride+col])-sum)>bounds[row*columns+col])throw std::runtime_error("Control failed FP64 reference bound");
                }
                for(size_t variant=1;variant<tiles.size();++variant) {
                    const auto& tile=tiles[variant];std::fill_n(static_cast<float*>(out.contents),rows*oStride,99.5f);run(tile);
                    auto actual=static_cast<float*>(out.contents);double squaredError=0,squaredValue=0;float maxError=0,maxValue=0;double worstForward=0;
                    for(uint64_t row=0;row<rows;++row)for(uint64_t col=0;col<oStride;++col) {
                        uint64_t index=row*oStride+col;if(col>=columns){if(actual[index]!=99.5f)throw std::runtime_error("Output padding overwritten");continue;}
                        if(!std::isfinite(actual[index]))throw std::runtime_error("Nonfinite tile result");
                        double forward=std::abs(double(actual[index])-oracle[row*columns+col])/bounds[row*columns+col];worstForward=std::max(worstForward,forward);if(forward>1)throw std::runtime_error("Candidate failed FP64 reference bound");
                        double error=double(actual[index])-reference[index];squaredError+=error*error;squaredValue+=double(reference[index])*reference[index];maxError=std::max(maxError,float(std::abs(error)));maxValue=std::max(maxValue,std::abs(reference[index]));
                    }
                    bool exact=std::memcmp(actual,reference.data(),reference.size()*4)==0;
                    std::printf("{\"kind\":\"check\",\"matrix\":\"%s\",\"rows\":%llu,\"variant\":\"%s\",\"bits_exact\":%s,\"max_abs_error\":%.9g,\"padding_preserved\":true,\"fp64_reference_passed\":true,\"worst_forward_error_ratio\":%.9g}\n",name.UTF8String,(unsigned long long)rows,tile.name,exact?"true":"false",maxError,worstForward);std::fflush(stdout);

                    for(int round=0;round<2;++round)for(bool candidate:{false,true,true,false}) {
                        const auto& selected=candidate?tile:tiles[0];for(int warm=0;warm<3;++warm)run(selected);std::vector<double> gpu,wall;
                        for(int trial=0;trial<12;++trial){auto timing=run(selected);gpu.push_back(timing.first);wall.push_back(timing.second);}
                        std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());
                        std::printf("{\"kind\":\"timing\",\"matrix\":\"%s\",\"rows\":%llu,\"variant\":\"%s\",\"round\":%d,\"policy\":\"%s\",\"gpu_ms_median\":%.9g,\"wall_ms_median\":%.9g}\n",name.UTF8String,(unsigned long long)rows,tile.name,round,candidate?"candidate":"control",(gpu[5]+gpu[6])*0.5,(wall[5]+wall[6])*0.5);std::fflush(stdout);
                    }
                }
            }
        }}return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"decode_tile_probe_failed: %s\n",error.what());return 1;}}
}
