// FP32-accumulated BF16 decode work-distribution experiment, separate from serving inference.
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
        std::vector<Tile> tiles={{"bf16_decode",1,16,4,128,nil},{"bf16_decode_sk2",1,16,2,128,nil},{"bf16_decode_sk4",1,16,4,128,nil}};
        auto reduce2=pipeline(device,library,"bf16_decode_sk2_reduce"),reduce4=pipeline(device,library,"bf16_decode_sk4_reduce");
        if(reduce2.maxTotalThreadsPerThreadgroup<64||reduce4.maxTotalThreadsPerThreadgroup<64)throw std::runtime_error("Reduction exceeds device limits");
        for(auto& tile:tiles){tile.pipeline=pipeline(device,library,tile.name);if(tile.pipeline.threadExecutionWidth!=32||tile.pipeline.maxTotalThreadsPerThreadgroup<tile.threads||tile.pipeline.staticThreadgroupMemoryLength>device.maxThreadgroupMemoryLength)throw std::runtime_error("Tile exceeds actual device limits");}
        NSString* folder=[NSString stringWithUTF8String:argv[2]];NSData* data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];
        NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        for(NSDictionary* item in manifest[@"cases"]) {@autoreleasepool {
            NSString* name=item[@"name"];uint64_t columns=[item[@"columns"] unsignedLongLongValue],inner=[item[@"inner"] unsignedLongLongValue];
            NSData* bytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:[name stringByAppendingString:@".bin"]]];
            if(bytes.length!=inner*columns*2)throw std::runtime_error("Incomplete BF16 matrix");
            auto weight=[device newBufferWithBytes:bytes.bytes length:bytes.length options:MTLResourceStorageModeShared];
            tiles[0].name=columns<=256?"bf16_decode_small":"bf16_decode";
            tiles[0].columns=columns<=256?4:16;tiles[0].threads=columns<=256?64:128;
            tiles[0].pipeline=pipeline(device,library,tiles[0].name);

            for(uint64_t rows:{1,2}) {
                uint64_t aStride=inner+8,oStride=columns+7;std::vector<float> input(rows*aStride,0);
                for(uint64_t row=0;row<rows;++row)for(uint64_t k=0;k<inner;++k)input[row*aStride+k]=std::sin(float(row*inner+k)*0.013f)*0.3f;
                auto a=[device newBufferWithBytes:input.data() length:input.size()*4 options:MTLResourceStorageModeShared];
                auto out=[device newBufferWithLength:rows*oStride*4 options:MTLResourceStorageModeShared];auto partials=[device newBufferWithLength:4*rows*oStride*4 options:MTLResourceStorageModeShared];Params p{rows,inner,columns,aStride,oStride};
                auto run=[&](const Tile& tile) {bool split=std::strstr(tile.name,"bf16_decode_sk")!=nullptr;
                    auto begin=std::chrono::steady_clock::now();auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];
                    [enc setComputePipelineState:tile.pipeline];[enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:weight offset:0 atIndex:1];[enc setBuffer:(split?partials:out) offset:0 atIndex:2];[enc setBytes:&p length:sizeof(p) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(((columns+tile.columns-1)/tile.columns)*(split?tile.inner:1),1,1) threadsPerThreadgroup:MTLSizeMake(tile.threads,1,1)];[enc endEncoding];
                    if(split){auto reduce=[cb computeCommandEncoder];[reduce setComputePipelineState:tile.inner==2?reduce2:reduce4];
                        [reduce setBuffer:partials offset:0 atIndex:0];[reduce setBuffer:out offset:0 atIndex:1];[reduce setBytes:&p length:sizeof(p) atIndex:2];
                        [reduce dispatchThreadgroups:MTLSizeMake((rows*columns+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[reduce endEncoding];}
                    finish(cb);
                    return std::pair<double,double>{(cb.GPUEndTime-cb.GPUStartTime)*1000,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()};
                };
                std::fill_n(static_cast<float*>(out.contents),rows*oStride,99.5f);run(tiles[0]);std::vector<float> reference(rows*oStride);std::memcpy(reference.data(),out.contents,reference.size()*4);
                std::vector<double> fp64(rows*columns),bounds(rows*columns);
                auto raw=static_cast<const uint16_t*>(bytes.bytes);
                const double epsilon=std::ldexp(1.0,-23),n=std::ceil(double(inner)/32)+32;
                const double gamma=n*epsilon/(1-n*epsilon);
                double controlRatio=0;
                for(uint64_t row=0;row<rows;++row)for(uint64_t col=0;col<columns;++col) {
                    double sum=0,absolute=0;
                    for(uint64_t k=0;k<inner;++k) {
                        uint32_t bits=uint32_t(raw[col*inner+k])<<16;float w;std::memcpy(&w,&bits,4);
                        double product=double(w)*double(input[row*aStride+k]);sum+=product;absolute+=std::abs(product);
                    }
                    auto index=row*columns+col;fp64[index]=sum;bounds[index]=gamma*absolute+1e-7;
                    if(!std::isfinite(sum)||!std::isfinite(absolute)||!std::isfinite(reference[row*oStride+col]))
                        throw std::runtime_error("Nonfinite independent reference/control");
                    controlRatio=std::max(controlRatio,std::abs(double(reference[row*oStride+col])-sum)/bounds[index]);
                }
                if(controlRatio>1)throw std::runtime_error("Control exceeds independent FP64 envelope");
                for(size_t variant=1;variant<tiles.size();++variant) {
                    const auto& tile=tiles[variant];std::fill_n(static_cast<float*>(out.contents),rows*oStride,99.5f);run(tile);
                    auto actual=static_cast<float*>(out.contents);double fp64Ratio=0,comparisonRatio=0,squaredError=0,squaredValue=0;float maxError=0,maxValue=0;
                    for(uint64_t row=0;row<rows;++row)for(uint64_t col=0;col<oStride;++col) {
                        uint64_t index=row*oStride+col;if(col>=columns){if(actual[index]!=99.5f)throw std::runtime_error("Output padding overwritten");continue;}
                        if(!std::isfinite(actual[index]))throw std::runtime_error("Nonfinite tile result");
                        double error=double(actual[index])-reference[index];
                        fp64Ratio=std::max(fp64Ratio,std::abs(double(actual[index])-fp64[row*columns+col])/bounds[row*columns+col]);
                        comparisonRatio=std::max(comparisonRatio,std::abs(error)/(5e-6+5e-5*std::abs(double(reference[index]))));squaredError+=error*error;squaredValue+=double(reference[index])*reference[index];maxError=std::max(maxError,float(std::abs(error)));maxValue=std::max(maxValue,std::abs(reference[index]));
                    }
                    if(fp64Ratio>1||comparisonRatio>1)throw std::runtime_error("Sub-SIMD exceeds unchanged numerical bounds");
                    bool exact=std::memcmp(actual,reference.data(),reference.size()*4)==0;
                    std::printf("{\"kind\":\"check\",\"matrix\":\"%s\",\"rows\":%llu,\"variant\":\"%s\",\"bits_exact\":%s,\"max_abs_error\":%.9g,\"fp64_worst_ratio\":%.9g,\"control_fp64_worst_ratio\":%.9g,\"comparison_worst_ratio\":%.9g,\"padding_preserved\":true}\n",name.UTF8String,(unsigned long long)rows,tile.name,exact?"true":"false",maxError,fp64Ratio,controlRatio,comparisonRatio);std::fflush(stdout);

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
