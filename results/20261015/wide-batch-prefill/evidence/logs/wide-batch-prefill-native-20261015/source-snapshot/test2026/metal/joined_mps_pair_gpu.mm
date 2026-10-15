// Exact-half gate/up row concatenation, single MPS projection and split rescale.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "../../src/backends/metal/exact_half_codec.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
struct Matrix {id<MTLBuffer> buffer;MPSMatrix* view;uint64_t stride;};
struct Params {uint64_t rows,columns,iStride,oStride;};
static Matrix allocate(id<MTLDevice> d,size_t rows,size_t cols,MPSDataType type) {
    size_t element=type==MPSDataTypeFloat16?2:4;
    auto bytes=[MPSMatrixDescriptor rowBytesFromColumns:cols dataType:type];
    auto descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:bytes dataType:type];
    auto buffer=[d newBufferWithLength:rows*bytes options:MTLResourceStorageModeShared];
    if(!buffer)throw std::runtime_error("Matrix allocation failed");std::memset(buffer.contents,0,rows*bytes);
    return {buffer,[[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor],bytes/element};
}
struct Weight {Matrix half;id<MTLBuffer> scales;uint64_t columns,inner;};
static Weight prepare(id<MTLDevice> device,NSString* folder,NSDictionary* item) {
    uint64_t inner=[item[@"inner"] unsignedLongLongValue],columns=[item[@"columns"] unsignedLongLongValue];
    NSString* file=[folder stringByAppendingPathComponent:[item[@"name"] stringByAppendingString:@".bin"]];
    auto data=[NSData dataWithContentsOfFile:file];if(data.length!=inner*columns*2)throw std::runtime_error("Weight byte count mismatch");
    auto raw=static_cast<const uint16_t*>(data.bytes);auto half=allocate(device,columns,inner,MPSDataTypeFloat16);std::vector<int32_t> scales(columns);
    for(uint64_t col=0;col<columns;++col) {
        uint16_t largest=0;for(uint64_t k=0;k<inner;++k)largest=std::max(largest,uint16_t(raw[col*inner+k]&0x7fff));
        if(!Garnet::ExactHalfRowScale(largest,scales[col]))throw std::runtime_error("Selected pair does not meet serving scale admission");
        for(uint64_t k=0;k<inner;++k)if(!Garnet::EncodeExactScaledHalf(raw[col*inner+k],scales[col],static_cast<uint16_t*>(half.buffer.contents)[col*half.stride+k]))throw std::runtime_error("Selected pair is not exact-half representable");
    }
    auto scaleBuffer=[device newBufferWithBytes:scales.data() length:scales.size()*4 options:MTLResourceStorageModeShared];
    if(!scaleBuffer)throw std::runtime_error("Scale allocation failed");return {half,scaleBuffer,columns,inner};
}
static void complete(id<MTLCommandBuffer> cb) {
    [cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cb.error.localizedDescription.UTF8String);
}
int main(int argc,char** argv) {@autoreleasepool {try {
    if(argc!=2)throw std::runtime_error("Expected pair assets");
    auto device=MTLCreateSystemDefaultDevice();auto queue0=[device newCommandQueue];
    if(!device||!queue0)throw std::runtime_error("Metal queues unavailable");NSError* error=nil;
    auto library=[device newLibraryWithSource:[NSString stringWithUTF8String:R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params {ulong rows,columns,iStride,oStride;};
kernel void rescale(device const float* input [[buffer(0)]],device const int* scales [[buffer(1)]],device float* output [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.rows*p.columns)return;ulong row=id/p.columns,col=id%p.columns;
 output[row*p.oStride+col]=ldexp(input[row*p.iStride+col],-scales[col]);
}
kernel void split_rescale(device const float* input [[buffer(0)]],device const int* scales [[buffer(1)]],device float* gate [[buffer(2)]],device float* up [[buffer(3)]],constant Params& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.rows*p.columns*2)return;ulong row=id/(p.columns*2),col=id%(p.columns*2);float value=ldexp(input[row*p.iStride+col],-scales[col]);
 if(col<p.columns)gate[row*p.oStride+col]=value;else up[row*p.oStride+col-p.columns]=value;
}
kernel void produce(device const float* source [[buffer(0)]],device float* target [[buffer(1)]],constant Params& p [[buffer(2)]],constant float& sign [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.rows*p.columns)return;ulong row=id/p.columns,col=id%p.columns;target[row*p.oStride+col]=source[row*p.iStride+col]*sign;
}
kernel void consume(device const float* gate [[buffer(0)]],device const float* up [[buffer(1)]],device float* output [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.rows*p.columns)return;ulong row=id/p.columns,col=id%p.columns;float g=gate[row*p.iStride+col],u=up[row*p.iStride+col];output[row*p.oStride+col]=g/(1.0f+exp(-g))*u;
}
)METAL"] options:nil error:&error];
    if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
    auto rescale=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rescale"] error:&error];
    if(!rescale)throw std::runtime_error(error.localizedDescription.UTF8String);
    auto split=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"split_rescale"] error:&error];
    if(!split)throw std::runtime_error("Split-rescale pipeline creation failed");
    auto producer=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"produce"] error:&error];
    auto consumer=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"consume"] error:&error];
    if(!producer||!consumer)throw std::runtime_error("Dependency pipeline creation failed");
    NSString* folder=[NSString stringWithUTF8String:argv[1]];
    auto manifestData=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];
    auto manifest=[NSJSONSerialization JSONObjectWithData:manifestData options:0 error:&error];
    NSDictionary* gate=nil;NSDictionary* up=nil;
    for(NSDictionary* item in manifest[@"cases"]) {
        if([item[@"tensor"] hasSuffix:@"mlp.gate_proj.weight"])gate=item;
        if([item[@"tensor"] hasSuffix:@"mlp.up_proj.weight"])up=item;
    }
    if(!gate||!up)throw std::runtime_error("Missing real gate/up pair");
    Weight weights[2]={prepare(device,folder,gate),prepare(device,folder,up)};
    if(weights[0].inner!=weights[1].inner||weights[0].columns!=weights[1].columns)throw std::runtime_error("Gate/up shapes differ");
    auto inner=weights[0].inner,columns=weights[0].columns;
    auto joinedWeight=allocate(device,columns*2,inner,MPSDataTypeFloat16);
    auto joinedScales=[device newBufferWithLength:columns*8 options:MTLResourceStorageModeShared];
    if(!joinedScales)throw std::runtime_error("Joined scales allocation failed");
    for(int i=0;i<2;++i)for(uint64_t col=0;col<columns;++col) {
        auto target=static_cast<uint16_t*>(joinedWeight.buffer.contents)+(i*columns+col)*joinedWeight.stride;
        auto original=static_cast<const uint16_t*>(weights[i].half.buffer.contents)+col*weights[i].half.stride;
        std::memcpy(target,original,inner*2);if(std::memcmp(target,original,inner*2))throw std::runtime_error("Joined weight bits changed");
        static_cast<int32_t*>(joinedScales.contents)[i*columns+col]=static_cast<const int32_t*>(weights[i].scales.contents)[col];
    }
    for(uint64_t rows:{22,24,26,32,44,52,64,80}) {@autoreleasepool {
        auto sourceInput=allocate(device,rows,inner,MPSDataTypeFloat32),input=allocate(device,rows,inner,MPSDataTypeFloat32),consumed=allocate(device,rows,columns,MPSDataTypeFloat32);Matrix temporary[2],output[2];MPSMatrixMultiplication* products[2];
        for(uint64_t row=0;row<rows;++row)for(uint64_t k=0;k<inner;++k)static_cast<float*>(sourceInput.buffer.contents)[row*sourceInput.stride+k]=std::sin(float(row*inner+k)*.013f)*.3f;
        for(int i=0;i<2;++i) {
            temporary[i]=allocate(device,rows,columns,MPSDataTypeFloat32);output[i]=allocate(device,rows,columns,MPSDataTypeFloat32);
            products[i]=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:columns interiorColumns:inner alpha:1 beta:0];
        }
        auto joinedTemporary=allocate(device,rows,columns*2,MPSDataTypeFloat32);
        auto joinedProduct=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:columns*2 interiorColumns:inner alpha:1 beta:0];
        auto encodeJoined=[&](id<MTLCommandBuffer> cb) {
            [joinedProduct encodeToCommandBuffer:cb leftMatrix:input.view rightMatrix:joinedWeight.view resultMatrix:joinedTemporary.view];
            auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:split];[enc setBuffer:joinedTemporary.buffer offset:0 atIndex:0];[enc setBuffer:joinedScales offset:0 atIndex:1];[enc setBuffer:output[0].buffer offset:0 atIndex:2];[enc setBuffer:output[1].buffer offset:0 atIndex:3];Params p{rows,columns,joinedTemporary.stride,output[0].stride};[enc setBytes:&p length:sizeof(p) atIndex:4];[enc dispatchThreads:MTLSizeMake(rows*columns*2,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];
        };
        auto encode=[&](id<MTLCommandBuffer> cb,int i) {
            [products[i] encodeToCommandBuffer:cb leftMatrix:input.view rightMatrix:weights[i].half.view resultMatrix:temporary[i].view];
            auto encoder=[cb computeCommandEncoder];[encoder setComputePipelineState:rescale];[encoder setBuffer:temporary[i].buffer offset:0 atIndex:0];[encoder setBuffer:weights[i].scales offset:0 atIndex:1];[encoder setBuffer:output[i].buffer offset:0 atIndex:2];
            Params p{rows,columns,temporary[i].stride,output[i].stride};[encoder setBytes:&p length:sizeof(p) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(rows*columns,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
        };
        auto encodeProducer=[&](id<MTLCommandBuffer> cb,float sign) {
            auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:producer];[enc setBuffer:sourceInput.buffer offset:0 atIndex:0];[enc setBuffer:input.buffer offset:0 atIndex:1];Params p{rows,inner,sourceInput.stride,input.stride};[enc setBytes:&p length:sizeof(p) atIndex:2];[enc setBytes:&sign length:sizeof(sign) atIndex:3];[enc dispatchThreads:MTLSizeMake(rows*inner,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];
        };
        auto encodeConsumer=[&](id<MTLCommandBuffer> cb) {
            auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:consumer];[enc setBuffer:output[0].buffer offset:0 atIndex:0];[enc setBuffer:output[1].buffer offset:0 atIndex:1];[enc setBuffer:consumed.buffer offset:0 atIndex:2];Params p{rows,columns,output[0].stride,consumed.stride};[enc setBytes:&p length:sizeof(p) atIndex:3];[enc dispatchThreads:MTLSizeMake(rows*columns,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];
        };
        auto run=[&](bool joined,float sign=1.0f) {
            auto begin=std::chrono::steady_clock::now();auto cb=[queue0 commandBuffer];encodeProducer(cb,sign);
            if(joined)encodeJoined(cb);else {encode(cb,0);encode(cb,1);}encodeConsumer(cb);
            [cb commit];complete(cb);double start=cb.GPUStartTime,end=cb.GPUEndTime;
            if(!(end>start&&start>0))throw std::runtime_error("GPU timestamps unavailable");
            return std::vector<double>{(end-start)*1000,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count(),0};
        };
        std::vector<float> reference[2];
        for(int i=0;i<2;++i)std::fill_n(static_cast<float*>(output[i].buffer.contents),rows*output[i].stride,99.5f);
        std::fill_n(static_cast<float*>(consumed.buffer.contents),rows*consumed.stride,99.5f);
        run(false);
        std::vector<float> consumerReference(rows*consumed.stride);std::memcpy(consumerReference.data(),consumed.buffer.contents,consumerReference.size()*4);
        for(int i=0;i<2;++i){reference[i].resize(rows*output[i].stride);std::memcpy(reference[i].data(),output[i].buffer.contents,reference[i].size()*4);std::fill_n(static_cast<float*>(output[i].buffer.contents),reference[i].size(),99.5f);}
        std::fill_n(static_cast<float*>(consumed.buffer.contents),rows*consumed.stride,99.5f);
        run(true);double worst=0,peak=0;bool bitsExact=true;
        for(int i=0;i<2;++i) {
            auto actual=static_cast<const float*>(output[i].buffer.contents);bitsExact&=std::memcmp(actual,reference[i].data(),reference[i].size()*4)==0;
            for(uint64_t row=0;row<rows;++row)for(uint64_t col=0;col<output[i].stride;++col) {
                auto index=row*output[i].stride+col;if(col>=columns){if(actual[index]!=99.5f||reference[i][index]!=99.5f)throw std::runtime_error("Padding overwritten");continue;}
                if(!std::isfinite(actual[index])||!std::isfinite(reference[i][index]))throw std::runtime_error("Nonfinite pair output");
                double difference=std::abs(double(actual[index])-reference[i][index]);peak=std::max(peak,difference);worst=std::max(worst,difference/(5e-6+5e-5*std::abs(double(reference[i][index]))));
            }
        }
        bool consumerBitsExact=std::memcmp(consumed.buffer.contents,consumerReference.data(),consumerReference.size()*4)==0;
        double consumerWorst=0;
        auto checkConsumer=[&]() {
            auto actual=static_cast<const float*>(consumed.buffer.contents);
            for(uint64_t row=0;row<rows;++row)for(uint64_t col=0;col<consumed.stride;++col) {
                uint64_t id=row*consumed.stride+col;
                if(col>=columns){if(actual[id]!=99.5f||consumerReference[id]!=99.5f)throw std::runtime_error("Consumer padding changed");continue;}
                if(!std::isfinite(actual[id])||!std::isfinite(consumerReference[id]))throw std::runtime_error("Nonfinite consumer");
                consumerWorst=std::max(consumerWorst,std::abs(double(actual[id])-consumerReference[id])/(5e-6+5e-5*std::abs(double(consumerReference[id]))));
            }
            if(consumerWorst>1)throw std::runtime_error("Consumer bound failed");
        };
        checkConsumer();
        // Alternate GPU-produced input data and compare each complete generation.
        for(int generation=0;generation<16;++generation) {
            float sign=generation%2?-1.0f:1.0f;run(false,sign);
            std::memcpy(consumerReference.data(),consumed.buffer.contents,consumerReference.size()*4);
            run(true,sign);
            consumerBitsExact&=std::memcmp(consumed.buffer.contents,consumerReference.data(),consumerReference.size()*4)==0;checkConsumer();
        }
        printf("{\"kind\":\"check\",\"rows\":%llu,\"columns\":%llu,\"inner\":%llu,\"bits_exact\":%s,\"padding_preserved\":true,\"dependent_consumer_bits_exact\":%s,\"consumer_worst_ratio\":%.9g,\"generation_checks\":16,\"worst_ratio\":%.9g,\"max_abs_error\":%.9g}\n",(unsigned long long)rows,(unsigned long long)columns,(unsigned long long)inner,bitsExact?"true":"false",consumerBitsExact?"true":"false",consumerWorst,worst,peak);fflush(stdout);
        if(worst>1)throw std::runtime_error("Pair output bound failed");
        for(int round=0;round<2;++round)for(bool joined:{false,true,true,false}) {
            for(int warm=0;warm<3;++warm)run(joined);std::vector<double> span,wall,overlap;
            for(int trial=0;trial<12;++trial){auto t=run(joined);span.push_back(t[0]);wall.push_back(t[1]);overlap.push_back(t[2]);}
            std::sort(span.begin(),span.end());std::sort(wall.begin(),wall.end());std::sort(overlap.begin(),overlap.end());
            printf("{\"kind\":\"timing\",\"rows\":%llu,\"round\":%d,\"joined\":%s,\"gpu_completion_span_ms\":%.9g,\"host_wall_ms\":%.9g,\"interval_overlap_ms\":%.9g}\n",(unsigned long long)rows,round,joined?"true":"false",(span[5]+span[6])*.5,(wall[5]+wall[6])*.5,(overlap[5]+overlap[6])*.5);fflush(stdout);
        }
    }}return 0;
}catch(const std::exception& e){fprintf(stderr,"joined_mps_pair_failed: %s\n",e.what());return 1;}}}
