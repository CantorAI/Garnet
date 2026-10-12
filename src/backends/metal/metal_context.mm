// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace Garnet {
struct MetalContext::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
};
static size_t Elements(size_t rows, size_t columns) {
    if (!rows || !columns || rows > std::numeric_limits<size_t>::max()/columns ||
        rows*columns > std::numeric_limits<size_t>::max()/sizeof(float))
        throw std::invalid_argument("Invalid Metal matrix dimensions");
    return rows*columns;
}
MetalContext::MetalContext(): impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        impl_->device=MTLCreateSystemDefaultDevice();
        if (!impl_->device || !MPSSupportsMTLDevice(impl_->device))
            throw std::runtime_error("No MPS-capable Metal GPU available");
        impl_->queue=[impl_->device newCommandQueue];
        if (!impl_->queue) throw std::runtime_error("Cannot create Metal command queue");
    }
}
MetalContext::~MetalContext() = default;
std::string MetalContext::DeviceName() const {
    return std::string([impl_->device.name UTF8String]);
}
std::vector<float> MetalContext::MatMul(const std::vector<float>& left,
    const std::vector<float>& right,size_t rows,size_t inner,size_t columns,
    bool transposeRight) {
    const auto lcount=Elements(rows,inner), rcount=Elements(inner,columns);
    const auto count=Elements(rows,columns);
    if(left.size()!=lcount || right.size()!=rcount)
        throw std::invalid_argument("Metal matrix input size mismatch");
    @autoreleasepool {
        auto matrix=[&](const float* source,size_t r,size_t c) -> MPSMatrix* {
            const auto stride=[MPSMatrixDescriptor rowBytesFromColumns:c dataType:MPSDataTypeFloat32];
            if(stride>std::numeric_limits<size_t>::max()/r)
                throw std::overflow_error("Metal matrix storage overflow");
            id<MTLBuffer> buffer=[impl_->device newBufferWithLength:r*stride
                options:MTLResourceStorageModeShared];
            if(!buffer) throw std::runtime_error("Metal matrix allocation failed");
            if(source) for(size_t i=0;i<r;++i)
                std::memcpy(static_cast<char*>(buffer.contents)+i*stride,source+i*c,c*sizeof(float));
            auto descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:r columns:c
                rowBytes:stride dataType:MPSDataTypeFloat32];
            return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
        };
        MPSMatrix* a=matrix(left.data(),rows,inner);
        MPSMatrix* b=matrix(right.data(),transposeRight?columns:inner,transposeRight?inner:columns);
        MPSMatrix* c=matrix(nullptr,rows,columns);
        auto op=[[MPSMatrixMultiplication alloc] initWithDevice:impl_->device
            transposeLeft:NO transposeRight:transposeRight resultRows:rows
            resultColumns:columns interiorColumns:inner alpha:1.0 beta:0.0];
        id<MTLCommandBuffer> command=[impl_->queue commandBuffer];
        if(!command) throw std::runtime_error("Cannot create Metal command buffer");
        [op encodeToCommandBuffer:command leftMatrix:a rightMatrix:b resultMatrix:c];
        [command commit];
        [command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)
            throw std::runtime_error(command.error ? std::string(command.error.localizedDescription.UTF8String)
                : "Metal execution did not complete");
        std::vector<float> output(count);
        for(size_t i=0;i<rows;++i)
            std::memcpy(output.data()+i*columns,static_cast<char*>(c.data.contents)+i*c.rowBytes,
                columns*sizeof(float));
        return output;
    }
}
}
