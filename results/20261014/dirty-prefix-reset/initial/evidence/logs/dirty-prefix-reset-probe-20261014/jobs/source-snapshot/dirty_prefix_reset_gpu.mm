// Component only: prior zero state, bounded GPU writes, then exact full reset.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using Clock=std::chrono::steady_clock;
struct Matrix {id<MTLBuffer> data;size_t rows,columns,stride,groups,groupColumns,dirtyColumns;};
static void need(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static void emit(NSDictionary* value){auto data=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];need(data!=nil,"JSON failed");puts([[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding].UTF8String);fflush(stdout);}
static void finish(id<MTLCommandBuffer> c){[c commit];[c waitUntilCompleted];need(c.status==MTLCommandBufferStatusCompleted,"GPU command failed");}
static void clear(const std::vector<Matrix>& matrices,int policy){
 for(const auto& m:matrices){
  std::vector<float> zeros;if(policy==0)zeros.resize(m.rows*m.columns,0.0f);
  for(size_t r=0;r<m.rows;++r){auto* row=static_cast<char*>(m.data.contents)+r*m.stride;
   if(policy==0)std::memcpy(row,zeros.data()+r*m.columns,m.columns*4);
   else if(policy==1)std::memset(row,0,m.columns*4);
   else for(size_t g=0;g<m.groups;++g)std::memset(row+g*m.groupColumns*4,0,m.dirtyColumns*4);
  }
 }
}
static void condition(const std::vector<Matrix>& matrices,id<MTLCommandQueue> queue,id<MTLComputePipelineState> pipeline,uint64_t generation){
 @autoreleasepool{auto c=[queue commandBuffer];auto e=[c computeCommandEncoder];need(c&&e,"Conditioner unavailable");[e setComputePipelineState:pipeline];
 for(const auto& m:matrices){if(!m.dirtyColumns)continue;
  uint64_t p[]={m.rows,m.columns,m.stride/4,m.groupColumns,m.dirtyColumns,generation};
  [e setBuffer:m.data offset:0 atIndex:0];[e setBytes:p length:sizeof(p) atIndex:1];
  [e dispatchThreads:MTLSizeMake(m.rows*m.columns,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 }[e endEncoding];finish(c);}
}
static void verify(const std::vector<Matrix>& matrices,id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLComputePipelineState> pipeline){
 @autoreleasepool{auto flag=[device newBufferWithLength:4 options:MTLResourceStorageModeShared];need(flag!=nil,"Verifier flag unavailable");*static_cast<uint32_t*>(flag.contents)=0;
 auto c=[queue commandBuffer];auto e=[c computeCommandEncoder];need(c&&e,"Verifier unavailable");[e setComputePipelineState:pipeline];[e setBuffer:flag offset:0 atIndex:2];
 for(const auto& m:matrices){auto bytes=static_cast<const uint8_t*>(m.data.contents);
  for(size_t r=0;r<m.rows;++r){for(size_t col=0;col<m.columns;++col){uint32_t bits;std::memcpy(&bits,bytes+r*m.stride+col*4,4);need(bits==0,"Full CPU positive-zero check failed");}
   for(size_t b=m.columns*4;b<m.stride;++b)need(bytes[r*m.stride+b]==0xa5,"Padding changed");}
  uint64_t p[]={m.rows,m.columns,m.stride/4,m.groupColumns,m.dirtyColumns,0};
  [e setBuffer:m.data offset:0 atIndex:0];[e setBytes:p length:sizeof(p) atIndex:1];
  [e dispatchThreads:MTLSizeMake(m.rows*m.columns,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 }[e endEncoding];finish(c);need(*static_cast<uint32_t*>(flag.contents)==0,"Full GPU positive-zero check failed");}
}
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==2,"Expected model profile JSON");auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];need(device&&queue,"Metal unavailable");
 NSString* source=[NSString stringWithUTF8String:R"METAL(
#include <metal_stdlib>
using namespace metal;
struct P{ulong rows,columns,stride,groupColumns,dirtyColumns,generation;};
kernel void dirty_state(device uint* data [[buffer(0)]],constant P& p [[buffer(1)]],uint id [[thread_position_in_grid]]){
 if(ulong(id)>=p.rows*p.columns)return;ulong col=ulong(id)%p.columns;
 if(col%p.groupColumns<p.dirtyColumns)data[(ulong(id)/p.columns)*p.stride+col]=0x3f000000u+uint((ulong(id)+p.generation*65537)%0x007fffffu);
}
kernel void verify_zero(device const uint* data [[buffer(0)]],constant P& p [[buffer(1)]],device atomic_uint* flag [[buffer(2)]],uint id [[thread_position_in_grid]]){
 if(ulong(id)<p.rows*p.columns&&data[(ulong(id)/p.columns)*p.stride+ulong(id)%p.columns]!=0)atomic_fetch_or_explicit(flag,1u,memory_order_relaxed);
}
)METAL"];
 NSError* error=nil;auto library=[device newLibraryWithSource:source options:nil error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto producer=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dirty_state"] error:&error];
 auto verifier=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"verify_zero"] error:&error];need(producer&&verifier,"Pipelines unavailable");
 auto profiles=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]] options:0 error:nil];need([profiles isKindOfClass:[NSArray class]],"Profiles unavailable");
 emit(@{@"kind":@"hardware",@"device":device.name});uint64_t generation=0;
 for(NSDictionary* profile in profiles){@autoreleasepool{std::vector<Matrix> matrices;size_t bytes=0,dirtyBytes=0;auto batch=[profile[@"batch"] unsignedLongLongValue];
  for(NSDictionary* s in profile[@"states"]){size_t columns=[s[@"columns"] unsignedLongLongValue],groups=[s[@"groups"] unsignedLongLongValue],groupColumns=[s[@"group_columns"] unsignedLongLongValue],dirtyColumns=[s[@"dirty_columns"] unsignedLongLongValue];
   need(batch&&columns&&groups&&groupColumns&&groups*groupColumns==columns&&dirtyColumns<=groupColumns&&batch*columns<=UINT32_MAX,"Invalid state bounds");
   size_t stride=[MPSMatrixDescriptor rowBytesFromColumns:columns dataType:MPSDataTypeFloat32];auto data=[device newBufferWithLength:batch*stride options:MTLResourceStorageModeShared];need(data!=nil,"State allocation failed");
   std::memset(data.contents,0xa5,batch*stride);matrices.push_back({data,batch,columns,stride,groups,groupColumns,dirtyColumns});bytes+=batch*columns*4;dirtyBytes+=batch*groups*dirtyColumns*4;
  }
  clear(matrices,1);verify(matrices,device,queue,verifier);
  // Induction: complete zero initialization, fresh bounded writes, complete zero after every policy.
  for(int policy=0;policy<3;++policy)for(int v=0;v<4;++v){condition(matrices,queue,producer,++generation);clear(matrices,policy);verify(matrices,device,queue,verifier);
   emit(@{@"kind":@"check",@"profile":profile[@"name"],@"policy":@(policy),@"generation":@(v),@"all_cpu_gpu_positive_zero":@YES,@"padding_preserved":@YES});}
  size_t warmCount=0;auto warm=Clock::now();do{for(int policy=0;policy<3;++policy){condition(matrices,queue,producer,++generation);clear(matrices,policy);++warmCount;}}while(ms(warm,Clock::now())<250);
  emit(@{@"kind":@"warmup",@"profile":profile[@"name"],@"elapsed_ms":@(ms(warm,Clock::now())),@"executions":@(warmCount)});
  for(int round=0;round<2;++round){int window=0;for(int policy:{0,1,2,2,1,0}){
   for(int w=0;w<3;++w){condition(matrices,queue,producer,++generation);clear(matrices,policy);}
   auto times=[NSMutableArray new];for(int sample=0;sample<12;++sample){condition(matrices,queue,producer,++generation);auto start=Clock::now();clear(matrices,policy);[times addObject:@(ms(start,Clock::now()))];}
   verify(matrices,device,queue,verifier);
   emit(@{@"kind":@"timing",@"profile":profile[@"name"],@"round":@(round),@"window":@(window++),@"policy":@(policy),@"logical_bytes":@(bytes),@"dirty_bytes":@(dirtyBytes),@"wall_ms":times,@"all_cpu_gpu_positive_zero":@YES,@"padding_preserved":@YES});
  }}
 }}return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}}
