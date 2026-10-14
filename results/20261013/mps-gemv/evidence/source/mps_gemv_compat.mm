// Isolated compatibility check: exact dyadic weights, FP32 vectors/output.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
int main(int argc,char** argv) {@autoreleasepool {try {
    if(argc!=2)throw std::runtime_error("Expected fp16 or fp32 matrix mode");
    bool half=std::strcmp(argv[1],"fp16")==0;
    if(!half&&std::strcmp(argv[1],"fp32")!=0)throw std::runtime_error("Unknown matrix mode");
    const NSUInteger rows=5,columns=37;
    auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];
    if(!device||!queue)throw std::runtime_error("Metal unavailable");
    auto type=half?MPSDataTypeFloat16:MPSDataTypeFloat32;
    auto stride=[MPSMatrixDescriptor rowBytesFromColumns:columns dataType:type];
    auto weight=[device newBufferWithLength:rows*stride options:MTLResourceStorageModeShared];
    auto input=[device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    auto output=[device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    if(!weight||!input||!output)throw std::runtime_error("Allocation failed");
    std::memset(weight.contents,0,rows*stride);std::memset(input.contents,0,256);std::memset(output.contents,0,256);
    for(NSUInteger k=0;k<columns;++k)static_cast<float*>(input.contents)[k]=(int(k%4)-2)*0.125f;
    for(NSUInteger row=0;row<rows;++row)for(NSUInteger k=0;k<columns;++k) {
        float w=(int((row+k)%7)-3)*0.125f;
        auto target=static_cast<char*>(weight.contents)+row*stride;
        if(half)reinterpret_cast<__fp16*>(target)[k]=__fp16(w);
        else reinterpret_cast<float*>(target)[k]=w;
    }
    auto matrix=[[MPSMatrix alloc] initWithBuffer:weight descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns rowBytes:stride dataType:type]];
    auto x=[[MPSVector alloc] initWithBuffer:input descriptor:[MPSVectorDescriptor vectorDescriptorWithLength:columns dataType:MPSDataTypeFloat32]];
    auto y=[[MPSVector alloc] initWithBuffer:output descriptor:[MPSVectorDescriptor vectorDescriptorWithLength:rows dataType:MPSDataTypeFloat32]];
    auto operation=[[MPSMatrixVectorMultiplication alloc] initWithDevice:device transpose:NO rows:rows columns:columns alpha:1 beta:0];
    if(!matrix||!x||!y||!operation)throw std::runtime_error("MPS GEMV object unavailable");
    std::printf("gemv_encode matrix=%s vector=fp32 output=fp32\n",argv[1]);std::fflush(stdout);
    auto command=[queue commandBuffer];[operation encodeToCommandBuffer:command inputMatrix:matrix inputVector:x resultVector:y];
    [command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("GEMV command failed");
    double maximum=0;
    for(NSUInteger row=0;row<rows;++row) {
        double expected=0;
        for(NSUInteger k=0;k<columns;++k)expected+=double((int((row+k)%7)-3)*0.125f)*double(static_cast<float*>(input.contents)[k]);
        float actual=static_cast<float*>(output.contents)[row];
        if(!std::isfinite(actual))throw std::runtime_error("Nonfinite GEMV output");
        maximum=std::max(maximum,std::abs(double(actual)-expected));
    }
    if(maximum!=0)throw std::runtime_error("Exact dyadic GEMV reference mismatch");
    std::printf("{\"matrix_dtype\":\"%s\",\"input_dtype\":\"fp32\",\"output_dtype\":\"fp32\",\"outputs\":5,\"reference_exact\":true}\n",argv[1]);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"gemv_compat_failed: %s\n",error.what());return 1;}}}
