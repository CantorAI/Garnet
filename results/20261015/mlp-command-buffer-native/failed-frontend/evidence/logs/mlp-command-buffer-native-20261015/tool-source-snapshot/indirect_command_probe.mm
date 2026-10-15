// Component gate only: fresh data, dependent FP32 dispatches, no model claim.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Params{uint32_t count;float add;};
int main(){@autoreleasepool{try{
 auto device=MTLCreateSystemDefaultDevice();check(device!=nil,"No Metal device");
 auto queue=[device newCommandQueue];NSError* error=nil;
 NSString* source=@"#include <metal_stdlib>\nusing namespace metal;\nstruct Params{uint count;float add;};\nkernel void step(device const float* a [[buffer(0)]],device float* b [[buffer(1)]],constant Params& p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p.count)b[i]=a[i]+p.add;}";
 MTLCompileOptions* opts=[MTLCompileOptions new];opts.fastMathEnabled=NO;
 auto library=[device newLibraryWithSource:source options:opts error:&error];check(library!=nil,"Shader compile failed");
 auto desc=[MTLComputePipelineDescriptor new];desc.computeFunction=[library newFunctionWithName:@"step"];desc.supportIndirectCommandBuffers=YES;
 auto pipeline=[device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&error];
 check(pipeline!=nil&&pipeline.supportIndirectCommandBuffers,"Indirect pipeline unavailable");
 std::cout<<"capability {\"device\":\""<<device.name.UTF8String<<"\",\"indirect_pipeline\":true,\"fast_math\":false}\n";
 size_t checks=0;
 for(size_t n:{size_t(896),size_t(1024),size_t(2048)})for(size_t depth:{size_t(16),size_t(64),size_t(256)}){@autoreleasepool{
  auto a=[device newBufferWithLength:n*4 options:MTLResourceStorageModeShared];
  auto b=[device newBufferWithLength:n*4 options:MTLResourceStorageModeShared];check(a&&b,"Data buffer allocation failed");
  auto parameters=[NSMutableArray new];
  auto icbDesc=[MTLIndirectCommandBufferDescriptor new];icbDesc.commandTypes=MTLIndirectCommandTypeConcurrentDispatchThreads;
  icbDesc.inheritBuffers=NO;icbDesc.inheritPipelineState=NO;icbDesc.maxKernelBufferBindCount=3;
  auto prepare=Clock::now();auto icb=[device newIndirectCommandBufferWithDescriptor:icbDesc maxCommandCount:depth options:MTLResourceStorageModeShared];
  check(icb!=nil,"Compute indirect buffer unavailable");
  for(size_t k=0;k<depth;++k){
   Params p{uint32_t(n),k%2?-.015625f:.03125f};auto param=[device newBufferWithBytes:&p length:sizeof(p) options:MTLResourceStorageModeShared];
   check(param!=nil,"Parameter allocation failed");[parameters addObject:param];
   auto command=[icb indirectComputeCommandAtIndex:k];[command setComputePipelineState:pipeline];
   [command setKernelBuffer:k%2?b:a offset:0 atIndex:0];[command setKernelBuffer:k%2?a:b offset:0 atIndex:1];
   [command setKernelBuffer:param offset:0 atIndex:2];
   [command concurrentDispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[command setBarrier];
  }
  std::cout<<"setup {\"elements\":"<<n<<",\"dispatches\":"<<depth<<",\"record_ms\":"<<ms(prepare,Clock::now())<<"}\n";
  size_t epoch=0;
  for(int round=0;round<2;++round)for(const char* mode:{"direct","indirect","indirect","direct","single","indirect","indirect","single"})for(int sample=-3;sample<12;++sample){@autoreleasepool{
   std::vector<float> expected(n);for(size_t i=0;i<n;++i)expected[i]=float(int((i+epoch)%17)-8)*.0625f;
   std::memcpy(a.contents,expected.data(),n*4);std::memset(b.contents,0,n*4);++epoch;
   for(size_t k=0;k<depth;++k)for(auto& value:expected)value+=k%2?-.015625f:.03125f;
   const auto begin=Clock::now();auto command=[queue commandBuffer];check(command!=nil,"Command buffer unavailable");
   if(std::strcmp(mode,"indirect")==0){
    auto encoder=[command computeCommandEncoder];check(encoder!=nil,"Indirect encoder unavailable");
    [encoder useResource:a usage:MTLResourceUsageRead|MTLResourceUsageWrite];[encoder useResource:b usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    for(id<MTLBuffer> param in parameters)[encoder useResource:param usage:MTLResourceUsageRead];
    [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0,depth)];[encoder endEncoding];
   }else{
    const bool single=std::strcmp(mode,"single")==0;id<MTLComputeCommandEncoder> encoder=single?[command computeCommandEncoder]:nil;
    for(size_t k=0;k<depth;++k){
     if(!single)encoder=[command computeCommandEncoder];check(encoder!=nil,"Direct encoder unavailable");
     if(single&&k)[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
     [encoder setComputePipelineState:pipeline];[encoder setBuffer:k%2?b:a offset:0 atIndex:0];[encoder setBuffer:k%2?a:b offset:0 atIndex:1];
     [encoder setBuffer:parameters[k] offset:0 atIndex:2];[encoder dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
     if(!single)[encoder endEncoding];
    }if(single)[encoder endEncoding];
   }
   const auto encoded=Clock::now();[command commit];[command waitUntilCompleted];const auto end=Clock::now();
   check(command.status==MTLCommandBufferStatusCompleted,"GPU command failed");
   check(std::memcmp(a.contents,expected.data(),n*4)==0,"Dependent output changed bits");++checks;
   if(sample>=0)std::cout<<"sample {\"mode\":\""<<mode<<"\",\"round\":"<<round<<",\"elements\":"<<n<<",\"dispatches\":"<<depth<<",\"sample\":"<<sample<<",\"encode_ms\":"<<ms(begin,encoded)<<",\"wall_ms\":"<<ms(begin,end)<<",\"gpu_ms\":"<<(command.GPUEndTime-command.GPUStartTime)*1000<<",\"exact\":true}\n";
  }}
 }}
 std::cout<<"completed {\"exact_checks\":"<<checks<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}}
