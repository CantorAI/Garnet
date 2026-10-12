// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace Garnet {
struct MetalMatrix::Impl {
    MPSMatrix* matrix;
    bool ready=false;
    bool pending=false;
};
struct MetalContext::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> command;
    std::vector<std::shared_ptr<MetalMatrix::Impl>> outputs;
    id<MTLComputePipelineState> rms;
    id<MTLComputePipelineState> silu;
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
        NSString* source=[NSString stringWithUTF8String:R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params { ulong rows,columns,aStride,bStride,cStride;float epsilon,offset; };
kernel void rms_norm(device const float* a [[buffer(0)]],device const float* weight [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint row [[thread_position_in_grid]]) {
    if(row>=p.rows)return;
    float sum=0;
    for(ulong j=0;j<p.columns;++j){float x=a[row*p.aStride+j];sum+=x*x;}
    float inverse=rsqrt(sum/float(p.columns)+p.epsilon);
    for(ulong j=0;j<p.columns;++j)c[row*p.cStride+j]=a[row*p.aStride+j]*inverse*(weight[j]+p.offset);
}
kernel void silu_mul(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    float gate=a[row*p.aStride+col];
    c[row*p.cStride+col]=(gate/(1.0f+exp(-gate)))*b[row*p.bStride+col];
}
)METAL"];
        NSError* error=nil;
        auto library=[impl_->device newLibraryWithSource:source options:nil error:&error];
        if(!library)throw std::runtime_error(std::string(error.localizedDescription.UTF8String));
        impl_->rms=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rms_norm"] error:&error];
        impl_->silu=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"silu_mul"] error:&error];
        if(!impl_->rms||!impl_->silu)throw std::runtime_error("Cannot create Qwen Metal pipelines");
    }
}
MetalContext::~MetalContext() { Abort(); }
std::string MetalContext::DeviceName() const {
    return std::string([impl_->device.name UTF8String]);
}
MetalMatrix MetalContext::Allocate(size_t rows,size_t columns) {
    Elements(rows,columns);
    const auto stride=[MPSMatrixDescriptor rowBytesFromColumns:columns dataType:MPSDataTypeFloat32];
    if(stride>std::numeric_limits<size_t>::max()/rows)
        throw std::overflow_error("Metal matrix storage overflow");
    id<MTLBuffer> buffer=[impl_->device newBufferWithLength:rows*stride options:MTLResourceStorageModeShared];
    if(!buffer)throw std::runtime_error("Metal matrix allocation failed");
    auto descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns
        rowBytes:stride dataType:MPSDataTypeFloat32];
    auto storage=std::make_shared<MetalMatrix::Impl>();
    storage->matrix=[[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
    return MetalMatrix(storage);
}
MetalMatrix MetalContext::Upload(const std::vector<float>& values,size_t rows,size_t columns) {
    if(values.size()!=Elements(rows,columns))throw std::invalid_argument("Metal matrix input size mismatch");
    auto result=Allocate(rows,columns);
    auto matrix=result.impl_->matrix;
    for(size_t i=0;i<rows;++i)
        std::memcpy(static_cast<char*>(matrix.data.contents)+i*matrix.rowBytes,
            values.data()+i*columns,columns*sizeof(float));
    result.impl_->ready=true;
    return result;
}
void MetalContext::Begin() {
    if(impl_->command)throw std::logic_error("Metal command batch already open");
    impl_->command=[impl_->queue commandBuffer];
    if(!impl_->command)throw std::runtime_error("Cannot create Metal command buffer");
}
void MetalContext::EncodeMatMul(const MetalMatrix& left,const MetalMatrix& right,
    const MetalMatrix& output,bool transposeRight) {
    if(!impl_->command)throw std::logic_error("Begin a Metal command batch first");
    if(!left.impl_||!right.impl_||!output.impl_)throw std::invalid_argument("Empty Metal matrix");
    auto a=left.impl_->matrix,b=right.impl_->matrix,c=output.impl_->matrix;
    if(a.device!=impl_->device||b.device!=impl_->device||c.device!=impl_->device)
        throw std::invalid_argument("Metal matrices belong to another device");
    auto usable=[&](const MetalMatrix& matrix){
        return matrix.impl_->ready || (matrix.impl_->pending &&
            std::find(impl_->outputs.begin(),impl_->outputs.end(),matrix.impl_)!=impl_->outputs.end());
    };
    if(!usable(left)||!usable(right))throw std::logic_error("Uninitialized Metal input");
    if(a.columns!=(transposeRight?b.columns:b.rows)||c.rows!=a.rows||
        c.columns!=(transposeRight?b.rows:b.columns))throw std::invalid_argument("Metal matmul shape mismatch");
    if(c.data==a.data||c.data==b.data)throw std::invalid_argument("In-place Metal matmul is unsupported");
    if(output.impl_->pending && std::find(impl_->outputs.begin(),impl_->outputs.end(),output.impl_)==impl_->outputs.end())
        throw std::logic_error("Metal output is in another command batch");
    auto op=[[MPSMatrixMultiplication alloc] initWithDevice:impl_->device transposeLeft:NO
        transposeRight:transposeRight resultRows:c.rows resultColumns:c.columns
        interiorColumns:a.columns alpha:1.0 beta:0.0];
    [op encodeToCommandBuffer:impl_->command leftMatrix:a rightMatrix:b resultMatrix:c];
    output.impl_->ready=false;output.impl_->pending=true;
    impl_->outputs.push_back(output.impl_);
}
void MetalContext::EncodePointwise(const MetalMatrix& left,const MetalMatrix& right,
    const MetalMatrix& output,bool rms,float epsilon,float offset) {
    if(!impl_->command)throw std::logic_error("Begin a Metal command batch first");
    if(!left.impl_||!right.impl_||!output.impl_)throw std::invalid_argument("Empty Metal matrix");
    auto a=left.impl_->matrix,b=right.impl_->matrix,c=output.impl_->matrix;
    if(a.device!=impl_->device||b.device!=impl_->device||c.device!=impl_->device)
        throw std::invalid_argument("Metal matrices belong to another device");
    auto usable=[&](const MetalMatrix& matrix){return matrix.impl_->ready ||
        (matrix.impl_->pending && std::find(impl_->outputs.begin(),impl_->outputs.end(),matrix.impl_)!=impl_->outputs.end());};
    if(!usable(left)||!usable(right))throw std::logic_error("Uninitialized Metal input");
    if(a.rows!=c.rows||a.columns!=c.columns||b.columns!=a.columns||b.rows!=(rms?1:a.rows))
        throw std::invalid_argument("Qwen Metal operation shape mismatch");
    if(rms && (!(epsilon>0)||!std::isfinite(epsilon)||!std::isfinite(offset)))
        throw std::invalid_argument("Invalid RMS normalization parameters");
    if(c.data==a.data||c.data==b.data)throw std::invalid_argument("In-place operation is unsupported");
    struct Params {uint64_t rows,columns,aStride,bStride,cStride;float epsilon,offset;};
    Params p{a.rows,a.columns,a.rowBytes/4,b.rowBytes/4,c.rowBytes/4,epsilon,offset};
    auto pipeline=rms?impl_->rms:impl_->silu;
    id<MTLComputeCommandEncoder> encoder=[impl_->command computeCommandEncoder];
    if(!encoder)throw std::runtime_error("Cannot create Metal compute encoder");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a.data offset:0 atIndex:0];[encoder setBuffer:b.data offset:0 atIndex:1];
    [encoder setBuffer:c.data offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];
    const size_t count=rms?a.rows:Elements(a.rows,a.columns);
    const size_t width=std::min(size_t(pipeline.maxTotalThreadsPerThreadgroup),count);
    [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
    [encoder endEncoding];output.impl_->ready=false;output.impl_->pending=true;
    impl_->outputs.push_back(output.impl_);
}
void MetalContext::EncodeRMSNorm(const MetalMatrix& input,const MetalMatrix& weight,
    const MetalMatrix& output,float epsilon,float weightOffset) {
    EncodePointwise(input,weight,output,true,epsilon,weightOffset);
}
void MetalContext::EncodeSiluMul(const MetalMatrix& gate,const MetalMatrix& up,const MetalMatrix& output) {
    EncodePointwise(gate,up,output,false,0,0);
}
void MetalContext::SubmitAndWait() {
    if(!impl_->command)throw std::logic_error("No Metal command batch to submit");
    auto command=impl_->command;
    [command commit];[command waitUntilCompleted];
    const bool success=command.status==MTLCommandBufferStatusCompleted;
    for(auto& output:impl_->outputs){output->pending=false;output->ready=success;}
    impl_->outputs.clear();impl_->command=nil;
    if(!success)throw std::runtime_error(command.error ?
        std::string(command.error.localizedDescription.UTF8String):"Metal execution did not complete");
}
void MetalContext::Abort() {
    for(auto& output:impl_->outputs){output->pending=false;output->ready=false;}
    impl_->outputs.clear();impl_->command=nil;
}
std::vector<float> MetalContext::Download(const MetalMatrix& value) const {
    if(!value.impl_||!value.impl_->ready||value.impl_->pending)
        throw std::logic_error("Metal matrix is not completed");
    auto matrix=value.impl_->matrix;
    if(matrix.device!=impl_->device)throw std::invalid_argument("Metal matrix belongs to another device");
    std::vector<float> result(Elements(matrix.rows,matrix.columns));
    for(size_t i=0;i<matrix.rows;++i)
        std::memcpy(result.data()+i*matrix.columns,static_cast<char*>(matrix.data.contents)+i*matrix.rowBytes,
            matrix.columns*sizeof(float));
    return result;
}
std::vector<float> MetalContext::MatMul(const std::vector<float>& left,
    const std::vector<float>& right,size_t rows,size_t inner,size_t columns,bool transposeRight) {
    auto a=Upload(left,rows,inner);
    auto b=Upload(right,transposeRight?columns:inner,transposeRight?inner:columns);
    auto c=Allocate(rows,columns);
    Begin();
    try {EncodeMatMul(a,b,c,transposeRight);SubmitAndWait();}
    catch(...) {Abort();throw;}
    return Download(c);
}
}
