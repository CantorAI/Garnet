// Experimental actual-QKV extended-scale projection versus FP32 MPS.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "experimental_extended_half_codec.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
struct Matrix {id<MTLBuffer> buffer;MPSMatrix* view;uint64_t stride;};
struct Params {uint64_t rows,columns,iStride,oStride;};
static Matrix matrix(id<MTLDevice> d,size_t rows,size_t cols,MPSDataType type) {
    auto bytes=[MPSMatrixDescriptor rowBytesFromColumns:cols dataType:type];
    auto descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:bytes dataType:type];
    auto buffer=[d newBufferWithLength:rows*bytes options:MTLResourceStorageModeShared];
    if(!buffer)throw std::runtime_error("Matrix allocation failed");std::memset(buffer.contents,0,rows*bytes);
    return {buffer,[[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor],bytes/(type==MPSDataTypeFloat16?2:4)};
}
int main(int argc,char** argv) {@autoreleasepool {try {
    if(argc!=2)throw std::runtime_error("Expected matrix assets");
    auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError* error=nil;
    auto library=[device newLibraryWithSource:[NSString stringWithUTF8String:R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params {ulong rows,columns,iStride,oStride;};
kernel void rescale(device const float* input [[buffer(0)]],device const int* scales [[buffer(1)]],
 device float* output [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.rows*p.columns)return;ulong row=id/p.columns,col=id%p.columns;
 output[row*p.oStride+col]=ldexp(input[row*p.iStride+col],-scales[col]);
}
)METAL"] options:nil error:&error];
    if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
    auto rescale=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rescale"] error:&error];
    if(!rescale)throw std::runtime_error(error.localizedDescription.UTF8String);
    NSString* folder=[NSString stringWithUTF8String:argv[1]];
    auto manifestData=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];
    auto manifest=[NSJSONSerialization JSONObjectWithData:manifestData options:0 error:&error];
    for(NSDictionary* item in manifest[@"cases"]) {@autoreleasepool {
        if(![item[@"tensor"] hasSuffix:@"linear_attn.in_proj_qkv.weight"])continue;
        NSString* name=item[@"name"];uint64_t inner=[item[@"inner"] unsignedLongLongValue],columns=[item[@"columns"] unsignedLongLongValue];
        auto bytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:[name stringByAppendingString:@".bin"]]];
        if(bytes.length!=inner*columns*2)throw std::runtime_error("BF16 byte count mismatch");
        auto raw=static_cast<const uint16_t*>(bytes.bytes);
        auto original=matrix(device,columns,inner,MPSDataTypeFloat16),transposed=matrix(device,columns,inner,MPSDataTypeFloat32);
        std::vector<int32_t> scales(columns);bool eligible=true;
        for(uint64_t col=0;col<columns&&eligible;++col) {
            uint16_t largest=0;for(uint64_t k=0;k<inner;++k)largest=std::max(largest,uint16_t(raw[col*inner+k]&0x7fff));
            if(!Garnet::ExactHalfRowScale(largest,scales[col])){eligible=false;break;}
            for(uint64_t k=0;k<inner;++k) {
                uint16_t encoded;if(!Garnet::EncodeExactScaledHalf(raw[col*inner+k],scales[col],encoded)){eligible=false;break;}
                static_cast<uint16_t*>(original.buffer.contents)[col*original.stride+k]=encoded;
                uint32_t floatBits=uint32_t(raw[col*inner+k])<<16;
                std::memcpy(static_cast<float*>(transposed.buffer.contents)+col*transposed.stride+k,&floatBits,4);
            }
        }
        if(!eligible){printf("{\"kind\":\"excluded\",\"name\":\"%s\",\"reason\":\"existing exact-half admission rejects matrix\"}\n",name.UTF8String);fflush(stdout);continue;}
        auto scaleBuffer=[device newBufferWithBytes:scales.data() length:scales.size()*4 options:MTLResourceStorageModeShared];
        for(uint64_t rows:{22,24,26,32,44,52,64,80}) {
            auto input=matrix(device,rows,inner,MPSDataTypeFloat32),temporary=matrix(device,rows,columns,MPSDataTypeFloat32),output=matrix(device,rows,columns,MPSDataTypeFloat32);
            for(uint64_t row=0;row<rows;++row)for(uint64_t k=0;k<inner;++k)static_cast<float*>(input.buffer.contents)[row*input.stride+k]=std::sin(float(row*inner+k)*.013f)*.3f;
            auto control=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:columns interiorColumns:inner alpha:1 beta:0];
            auto candidate=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:columns interiorColumns:inner alpha:1 beta:0];
            Params p{rows,columns,temporary.stride,output.stride};
            auto run=[&](bool useCandidate) {
                auto start=std::chrono::steady_clock::now();auto command=[queue commandBuffer];
                [(useCandidate?candidate:control) encodeToCommandBuffer:command leftMatrix:input.view rightMatrix:(useCandidate?original:transposed).view resultMatrix:(useCandidate?temporary:output).view];
                if(useCandidate) {
                auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:rescale];
                [encoder setBuffer:temporary.buffer offset:0 atIndex:0];[encoder setBuffer:scaleBuffer offset:0 atIndex:1];[encoder setBuffer:output.buffer offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];
                [encoder dispatchThreads:MTLSizeMake(rows*columns,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(uint64_t(128),rows*columns),1,1)];[encoder endEncoding];
                }
                [command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
                return std::pair<double,double>{(command.GPUEndTime-command.GPUStartTime)*1000,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()};
            };
            std::fill_n(static_cast<float*>(output.buffer.contents),rows*output.stride,99.5f);run(false);
            std::vector<float> reference(rows*output.stride);std::memcpy(reference.data(),output.buffer.contents,reference.size()*4);
            std::fill_n(static_cast<float*>(output.buffer.contents),rows*output.stride,99.5f);run(true);
            double worst=0,peak=0,tinyWorst=0;size_t tinyValues=0;auto actual=static_cast<const float*>(output.buffer.contents);
            for(uint64_t row=0;row<rows;++row)for(uint64_t col=0;col<output.stride;++col) {
                uint64_t index=row*output.stride+col;if(col>=columns){if(actual[index]!=99.5f||reference[index]!=99.5f)throw std::runtime_error("Output padding overwritten");continue;}
                if(!std::isfinite(actual[index])||!std::isfinite(reference[index]))throw std::runtime_error("Nonfinite projection");
                double difference=std::abs(double(actual[index])-reference[index]);
                if(scales[col]>64){++tinyValues;tinyWorst=std::max(tinyWorst,difference/(1e-40+5e-5*std::abs(double(reference[index]))));}
                peak=std::max(peak,difference);worst=std::max(worst,difference/(5e-6+5e-5*std::abs(double(reference[index]))));
            }
            printf("{\"kind\":\"check\",\"name\":\"%s\",\"rows\":%llu,\"extended_scale\":true,\"padding_preserved\":true,\"worst_ratio\":%.9g,\"max_abs_error\":%.9g}\n",name.UTF8String,(unsigned long long)rows,worst,peak);fflush(stdout);
            printf("{\"kind\":\"tiny_check\",\"rows\":%llu,\"values\":%zu,\"worst_ratio\":%.9g}\n",(unsigned long long)rows,tinyValues,tinyWorst);fflush(stdout);
            if(tinyWorst>1)throw std::runtime_error("Tiny projection changed beyond strict bound");
            if(worst>1)throw std::runtime_error("Projection exceeds unchanged output bound");
            for(int round=0;round<2;++round)for(bool useCandidate:{false,true,true,false}) {
                for(int warm=0;warm<3;++warm)run(useCandidate);std::vector<double> gpu,wall;
                for(int trial=0;trial<12;++trial){auto timing=run(useCandidate);gpu.push_back(timing.first);wall.push_back(timing.second);}std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());
                printf("{\"kind\":\"timing\",\"name\":\"%s\",\"rows\":%llu,\"round\":%d,\"candidate\":%s,\"gpu_ms\":%.9g,\"wall_ms\":%.9g}\n",name.UTF8String,(unsigned long long)rows,round,useCandidate?"true":"false",(gpu[5]+gpu[6])*.5,(wall[5]+wall[6])*.5);fflush(stdout);
            }
        }
    }}return 0;
}catch(const std::exception& error){fprintf(stderr,"extended_scale_projection_failed: %s\n",error.what());return 1;}}}
