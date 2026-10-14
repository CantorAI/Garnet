// Component experiment only: synchronous reset including GPU queue/wait cost.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
struct Matrix {id<MTLBuffer> data;size_t rows,columns,stride;};
static void finish(id<MTLCommandBuffer> command) {
    [command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted)
        throw std::runtime_error("GPU reset command failed");
}
static void clear(const std::vector<Matrix>& matrices,int policy,id<MTLCommandQueue> queue) {
    @autoreleasepool {
    if(policy<2) {
        for(const auto& m:matrices) {
            std::vector<float> zeros;
            if(policy==0)zeros.resize(m.rows*m.columns,0.0f);
            for(size_t row=0;row<m.rows;++row) {
                auto target=static_cast<char*>(m.data.contents)+row*m.stride;
                if(policy==0)std::memcpy(target,zeros.data()+row*m.columns,m.columns*4);
                else std::memset(target,0,m.columns*4);
            }
        }
    } else {
        auto command=[queue commandBuffer];auto encoder=[command blitCommandEncoder];
        if(!command||!encoder)throw std::runtime_error("GPU reset encoder unavailable");
        for(const auto& m:matrices)for(size_t row=0;row<m.rows;++row)
            [encoder fillBuffer:m.data range:NSMakeRange(row*m.stride,m.columns*4) value:0];
        [encoder endEncoding];finish(command);
    }
    }
}
static double median(std::vector<double> v) {
    std::sort(v.begin(),v.end());return (v[v.size()/2]+v[(v.size()-1)/2])/2;
}
int main(int argc,char** argv) {@autoreleasepool {try {
    if(argc!=2)throw std::runtime_error("Expected profile TSV");
    auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];
    if(!device||!queue)throw std::runtime_error("Metal unavailable");
    NSError* error=nil;
    NSString* shader=@"#include <metal_stdlib>\nusing namespace metal;\nstruct CheckParams {ulong rows,columns,stride;};\n"
        "kernel void verify_zero(device const uint* data [[buffer(0)]],device atomic_uint* flag [[buffer(1)]],"
        "constant CheckParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]) {"
        "if(ulong(id)<p.rows*p.columns&&data[(ulong(id)/p.columns)*p.stride+ulong(id)%p.columns]!=0)atomic_fetch_or_explicit(flag,1u,memory_order_relaxed);}";
    auto library=[device newLibraryWithSource:shader options:nil error:&error];
    if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
    auto kernel=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"verify_zero"] error:&error];
    if(!kernel)throw std::runtime_error("GPU verifier unavailable");
    std::ifstream input(argv[1]);if(!input)throw std::runtime_error("Profiles unavailable");
    std::string label;size_t batch,count,columns;
    while(input>>label>>batch>>count) {@autoreleasepool {
        std::vector<Matrix> matrices;size_t logicalBytes=0;
        for(size_t i=0;i<count;++i) {
            if(!(input>>columns)||!columns||batch*columns>UINT32_MAX)throw std::runtime_error("Invalid profile");
            size_t stride=[MPSMatrixDescriptor rowBytesFromColumns:columns dataType:MPSDataTypeFloat32];
            auto buffer=[device newBufferWithLength:batch*stride options:MTLResourceStorageModeShared];
            if(!buffer)throw std::runtime_error("Reset allocation failed");
            matrices.push_back({buffer,batch,columns,stride});logicalBytes+=batch*columns*4;
        }
        for(int policy=0;policy<3;++policy)for(int generation=0;generation<3;++generation) {
            for(const auto& m:matrices)std::memset(m.data.contents,0x5a+generation,m.rows*m.stride);
            clear(matrices,policy,queue);
            auto flag=[device newBufferWithLength:4 options:MTLResourceStorageModeShared];
            if(!flag)throw std::runtime_error("Verifier flag allocation failed");
            *static_cast<uint32_t*>(flag.contents)=0;
            auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
            [encoder setComputePipelineState:kernel];[encoder setBuffer:flag offset:0 atIndex:1];
            for(const auto& m:matrices) {
                const uint64_t p[3]={m.rows,m.columns,m.stride/4};
                [encoder setBuffer:m.data offset:0 atIndex:0];[encoder setBytes:p length:sizeof(p) atIndex:2];
                [encoder dispatchThreads:MTLSizeMake(m.rows*m.columns,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                auto bytes=static_cast<const uint8_t*>(m.data.contents);
                for(size_t row=0;row<m.rows;++row) {
                    for(size_t col=0;col<m.columns;++col) {
                        uint32_t bits;std::memcpy(&bits,bytes+row*m.stride+col*4,4);
                        if(bits!=0)throw std::runtime_error("CPU positive-zero check failed");
                    }
                    for(size_t offset=m.columns*4;offset<m.stride;++offset)
                        if(bytes[row*m.stride+offset]!=0x5a+generation)throw std::runtime_error("Padding changed");
                }
            }
            [encoder endEncoding];finish(command);
            if(*static_cast<uint32_t*>(flag.contents))throw std::runtime_error("GPU positive-zero check failed");
            std::printf("{\"kind\":\"check\",\"profile\":\"%s\",\"policy\":%d,\"generation\":%d,\"bytes\":%zu,\"cpu_gpu_positive_zero\":true,\"padding_preserved\":true}\n",label.c_str(),policy,generation,logicalBytes);
        }
        for(int round=0;round<2;++round)for(int policy:{0,1,2,2,1,0}) {
            for(const auto& m:matrices)std::memset(m.data.contents,0x5a,m.rows*m.stride);
            for(int warm=0;warm<3;++warm)clear(matrices,policy,queue);
            std::vector<double> times;
            for(int sample=0;sample<8;++sample) {
                auto start=std::chrono::steady_clock::now();clear(matrices,policy,queue);
                times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
            }
            std::printf("{\"kind\":\"timing\",\"profile\":\"%s\",\"policy\":%d,\"round\":%d,\"samples\":8,\"wall_ms_median\":%.9g}\n",label.c_str(),policy,round,median(times));
        }
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}}
