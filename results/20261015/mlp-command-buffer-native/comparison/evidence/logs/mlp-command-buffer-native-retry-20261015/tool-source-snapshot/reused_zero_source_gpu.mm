// Component only: include reset and a dependent GPU read in one observed wall clock.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
struct Matrix{id<MTLBuffer> data;size_t rows,columns,stride;};
static void need(bool b,const char* text){if(!b)throw std::runtime_error(text);}
static void finish(id<MTLCommandBuffer> cb){[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Metal command failed");}
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==2,"Expected actual-config profiles TSV");auto d=MTLCreateSystemDefaultDevice();auto queue=[d newCommandQueue];NSError* error=nil;
 NSString* code=@"#include <metal_stdlib>\nusing namespace metal;\nstruct P{ulong rows,columns,stride;};\n"
 "kernel void condition_state(device uint* a [[buffer(0)]],constant P& p [[buffer(1)]],uint id [[thread_position_in_grid]]){if(ulong(id)<p.rows*p.columns)a[(ulong(id)/p.columns)*p.stride+ulong(id)%p.columns]=0x5a5a5a5au;}\n"
 "kernel void consume_zero(device const uint* a [[buffer(0)]],device atomic_uint* flag [[buffer(1)]],constant P& p [[buffer(2)]],uint id [[thread_position_in_grid]]){if(ulong(id)<p.rows*p.columns&&a[(ulong(id)/p.columns)*p.stride+ulong(id)%p.columns]!=0)atomic_fetch_or_explicit(flag,1u,memory_order_relaxed);}";
 auto library=[d newLibraryWithSource:code options:nil error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto producer=[d newComputePipelineStateWithFunction:[library newFunctionWithName:@"condition_state"] error:&error],consumer=[d newComputePipelineStateWithFunction:[library newFunctionWithName:@"consume_zero"] error:&error];need(producer&&consumer,"Missing pipeline");
 auto flag=[d newBufferWithLength:4 options:MTLResourceStorageModeShared];std::ifstream input(argv[1]);std::string label;size_t batch,count,columns;
 while(input>>label>>batch>>count){@autoreleasepool{
  std::vector<Matrix> matrices;size_t bytes=0;
  for(size_t i=0;i<count;++i){need(bool(input>>columns)&&columns>0&&batch*columns<UINT32_MAX,"Invalid profile");size_t stride=[MPSMatrixDescriptor rowBytesFromColumns:columns dataType:MPSDataTypeFloat32];auto buffer=[d newBufferWithLength:batch*stride options:MTLResourceStorageModeShared];need(buffer!=nil,"Allocation failed");std::memset(buffer.contents,0x6b,batch*stride);matrices.push_back({buffer,batch,columns,stride});bytes+=batch*columns*4;}
  size_t maximum=0;for(const auto& m:matrices)if(m.rows*m.columns>maximum)maximum=m.rows*m.columns;
  std::vector<float> cachedZero(maximum,0.0f);
  auto condition=[&](){auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:producer];for(const auto& m:matrices){const uint64_t p[]={m.rows,m.columns,m.stride/4};[e setBuffer:m.data offset:0 atIndex:0];[e setBytes:p length:sizeof(p) atIndex:1];[e dispatchThreads:MTLSizeMake(m.rows*m.columns,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}[e endEncoding];finish(cb);};
  auto run=[&](int policy){*static_cast<uint32_t*>(flag.contents)=0;auto begin=std::chrono::steady_clock::now();
   if(policy<2)for(const auto& m:matrices){std::vector<float> zero;if(policy==0)zero.resize(m.rows*m.columns,0.f);for(size_t row=0;row<m.rows;++row){auto target=static_cast<char*>(m.data.contents)+row*m.stride;if(policy==0)std::memcpy(target,zero.data()+row*m.columns,m.columns*4);else std::memcpy(target,cachedZero.data()+row*m.columns,m.columns*4);}}
   auto cb=[queue commandBuffer];
   if(policy==2){auto e=[cb blitCommandEncoder];need(e!=nil,"Blit encoder missing");for(const auto& m:matrices)for(size_t row=0;row<m.rows;++row)[e fillBuffer:m.data range:NSMakeRange(row*m.stride,m.columns*4) value:0];[e endEncoding];}
   auto e=[cb computeCommandEncoder];[e setComputePipelineState:consumer];[e setBuffer:flag offset:0 atIndex:1];for(const auto& m:matrices){const uint64_t p[]={m.rows,m.columns,m.stride/4};[e setBuffer:m.data offset:0 atIndex:0];[e setBytes:p length:sizeof(p) atIndex:2];[e dispatchThreads:MTLSizeMake(m.rows*m.columns,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}[e endEncoding];finish(cb);
   double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count(),gpu=(cb.GPUEndTime-cb.GPUStartTime)*1000;need(*static_cast<uint32_t*>(flag.contents)==0,"Dependent GPU read saw nonzero bits");return std::pair<double,double>{wall,gpu};};
  for(int policy=0;policy<3;++policy)for(int generation=0;generation<3;++generation){condition();run(policy);for(const auto& m:matrices)for(size_t row=0;row<m.rows;++row){auto raw=static_cast<const unsigned char*>(m.data.contents)+row*m.stride;for(size_t col=0;col<m.columns;++col){uint32_t bits;std::memcpy(&bits,raw+col*4,4);need(bits==0,"CPU positive-zero failure");}for(size_t offset=m.columns*4;offset<m.stride;++offset)need(raw[offset]==0x6b,"Row padding modified");}std::printf("{\"kind\":\"check\",\"profile\":\"%s\",\"policy\":%d,\"generation\":%d,\"bytes\":%zu,\"positive_zero_cpu_gpu\":true,\"padding_preserved\":true}\n",label.c_str(),policy,generation,bytes);std::fflush(stdout);}
  auto warm=std::chrono::steady_clock::now();while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<.25)for(int policy=0;policy<3;++policy){condition();run(policy);}
  const int order[]={0,1,2,2,1,0};
  for(int round=0;round<2;++round)for(int window=0;window<6;++window){int policy=order[window];std::vector<double> wall,gpu;for(int trial=0;trial<8;++trial){condition();auto times=run(policy);wall.push_back(times.first);gpu.push_back(times.second);}std::printf("{\"kind\":\"timing\",\"profile\":\"%s\",\"round\":%d,\"window\":%d,\"policy\":%d,\"wall_ms\":[",label.c_str(),round,window,policy);for(size_t i=0;i<wall.size();++i)std::printf("%s%.17g",i?",":"",wall[i]);std::printf("],\"gpu_ms\":[");for(size_t i=0;i<gpu.size();++i)std::printf("%s%.17g",i?",":"",gpu[i]);std::puts("]}");std::fflush(stdout);}
 }}return 0;
}catch(const std::exception& e){std::fprintf(stderr,"reused_zero_source_failed: %s\n",e.what());return 1;}}}
