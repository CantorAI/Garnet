// Four original MLP kernels, with fresh input and bit-exact intermediate checks.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using Clock=std::chrono::steady_clock;
struct Projection{uint64_t rows,inner,columns,aStride,oStride;};
struct Pointwise{uint64_t rows,columns,aStride,bStride,cStride;float epsilon,offset;};
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
static void emit(NSDictionary* object){auto data=[NSJSONSerialization dataWithJSONObject:object options:0 error:nil];require(data!=nil,"JSON unavailable");std::printf("%s\n",[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding].UTF8String);std::fflush(stdout);}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device,id<MTLLibrary> library,NSString* name,bool recordable){
 auto descriptor=[MTLComputePipelineDescriptor new];descriptor.computeFunction=[library newFunctionWithName:name];descriptor.supportIndirectCommandBuffers=recordable;
 NSError* error=nil;auto p=[device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);return p;
}
int main(int argc,char** argv){@autoreleasepool{try{
 require(argc==3,"Expected shader and MLP manifest");NSError* error=nil;auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];
 auto shader=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];auto library=[device newLibraryWithSource:shader options:nil error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 id<MTLComputePipelineState> p[2][2]={{pipeline(device,library,@"bf16_decode",false),pipeline(device,library,@"silu_mul",false)},{pipeline(device,library,@"bf16_decode",true),pipeline(device,library,@"silu_mul",true)}};
 require(p[0][1].maxTotalThreadsPerThreadgroup==p[1][1].maxTotalThreadsPerThreadgroup,"Pipeline flag changes Silu launch width");
 auto json=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]];auto manifest=[NSJSONSerialization JSONObjectWithData:json options:0 error:&error];require(manifest!=nil,"MLP manifest unavailable");
 size_t executions=0;
 for(NSDictionary* item in manifest[@"models"]){@autoreleasepool{
  NSString* model=item[@"model"];const uint64_t hidden=[item[@"hidden"] unsignedLongLongValue],intermediate=[item[@"intermediate"] unsignedLongLongValue];
  NSData* raw[3];id<MTLBuffer> weights[3];NSString* names[3]={@"gate",@"up",@"down"};
  for(int i=0;i<3;++i){raw[i]=[NSData dataWithContentsOfFile:item[names[i]]];require(raw[i].length==hidden*intermediate*2,"MLP BF16 data size mismatch");weights[i]=[device newBufferWithBytes:raw[i].bytes length:raw[i].length options:MTLResourceStorageModeShared];require(weights[i]!=nil,"MLP weight allocation failed");}
  for(uint64_t rows:{1,2}){@autoreleasepool{
   const uint64_t hStride=((hidden+15)/16)*16+16,iStride=((intermediate+15)/16)*16+16;
   std::vector<float> base(rows*hStride,0);for(uint64_t row=0;row<rows;++row)for(uint64_t k=0;k<hidden;++k)base[row*hStride+k]=std::sin(float(row*hidden+k)*.013f)*.3f;
   auto input=[device newBufferWithLength:base.size()*4 options:MTLResourceStorageModeShared];id<MTLBuffer> values[4];
   for(int i=0;i<4;++i){values[i]=[device newBufferWithLength:rows*(i==3?hStride:iStride)*4 options:MTLResourceStorageModeShared];require(values[i]!=nil,"MLP activation allocation failed");}
   Projection gp{rows,hidden,intermediate,hStride,iStride},dp{rows,intermediate,hidden,iStride,hStride};Pointwise sp{rows,intermediate,iStride,iStride,iStride,0,0};
   id<MTLBuffer> parameter[3]={[device newBufferWithBytes:&gp length:sizeof(gp) options:MTLResourceStorageModeShared],[device newBufferWithBytes:&sp length:sizeof(sp) options:MTLResourceStorageModeShared],[device newBufferWithBytes:&dp length:sizeof(dp) options:MTLResourceStorageModeShared]};
   id<MTLBuffer> bindings[4][4]={{input,weights[0],values[0],parameter[0]},{input,weights[1],values[1],parameter[0]},{values[0],values[1],values[2],parameter[1]},{values[2],weights[2],values[3],parameter[2]}};
   const size_t widths[4]={128,128,std::min(uint64_t(p[0][1].maxTotalThreadsPerThreadgroup),rows*intermediate),128};
   MTLSize grids[4]={MTLSizeMake((intermediate+15)/16,1,1),MTLSizeMake((intermediate+15)/16,1,1),MTLSizeMake(rows*intermediate,1,1),MTLSizeMake((hidden+15)/16,1,1)};
   auto recordBegin=Clock::now();id<MTLIndirectCommandBuffer> recorded[2];
   for(int concurrent=0;concurrent<2;++concurrent){auto descriptor=[MTLIndirectCommandBufferDescriptor new];descriptor.commandTypes=MTLIndirectCommandTypeConcurrentDispatch|MTLIndirectCommandTypeConcurrentDispatchThreads;descriptor.inheritBuffers=NO;descriptor.inheritPipelineState=NO;descriptor.maxKernelBufferBindCount=4;
    recorded[concurrent]=[device newIndirectCommandBufferWithDescriptor:descriptor maxCommandCount:4 options:MTLResourceStorageModeShared];require(recorded[concurrent]!=nil,"MLP indirect command allocation failed");
    for(int op=0;op<4;++op){auto command=[recorded[concurrent] indirectComputeCommandAtIndex:op];
     // A barrier waits for prior commands BEFORE this command. Gate/up may
     // overlap, but both must finish before Silu, then Silu before down.
     if(!concurrent||op>=2)[command setBarrier];
     [command setComputePipelineState:p[1][op==2?1:0]];for(int slot=0;slot<4;++slot)[command setKernelBuffer:bindings[op][slot] offset:0 atIndex:slot];
     if(op==2)[command concurrentDispatchThreads:grids[op] threadsPerThreadgroup:MTLSizeMake(widths[op],1,1)];else [command concurrentDispatchThreadgroups:grids[op] threadsPerThreadgroup:MTLSizeMake(widths[op],1,1)];
    }
   }
   emit(@{@"kind":@"setup",@"model":model,@"rows":@(rows),@"record_two_plans_ms":@(ms(recordBegin,Clock::now()))});
   const float scales[4]={1,-1,.5f,2};std::vector<float> reference[4][4];size_t epoch=0;
   auto run=[&](int mode,int variant,bool validate){@autoreleasepool{
    for(size_t i=0;i<base.size();++i)static_cast<float*>(input.contents)[i]=base[i]*scales[variant];for(int i=0;i<4;++i)std::fill_n(static_cast<float*>(values[i].contents),rows*(i==3?hStride:iStride),99.5f);
    const auto begin=Clock::now();auto command=[queue commandBuffer];
    if(mode>=2){auto encoder=[command computeCommandEncoder];id<MTLResource> reads[7]={input,weights[0],weights[1],weights[2],parameter[0],parameter[1],parameter[2]};id<MTLResource> updates[4]={values[0],values[1],values[2],values[3]};
     [encoder useResources:reads count:7 usage:MTLResourceUsageRead];[encoder useResources:updates count:4 usage:MTLResourceUsageRead|MTLResourceUsageWrite];[encoder executeCommandsInBuffer:recorded[mode-2] withRange:NSMakeRange(0,4)];[encoder endEncoding];
    }else for(int op=0;op<4;++op){auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:p[mode][op==2?1:0]];for(int slot=0;slot<3;++slot)[encoder setBuffer:bindings[op][slot] offset:0 atIndex:slot];
     if(op==2)[encoder setBytes:&sp length:sizeof(sp) atIndex:3];else if(op==3)[encoder setBytes:&dp length:sizeof(dp) atIndex:3];else [encoder setBytes:&gp length:sizeof(gp) atIndex:3];
     if(op==2)[encoder dispatchThreads:grids[op] threadsPerThreadgroup:MTLSizeMake(widths[op],1,1)];else [encoder dispatchThreadgroups:grids[op] threadsPerThreadgroup:MTLSizeMake(widths[op],1,1)];[encoder endEncoding];
    }
    const auto encoded=Clock::now();[command commit];[command waitUntilCompleted];const auto end=Clock::now();require(command.status==MTLCommandBufferStatusCompleted,"MLP GPU command failed");
    if(validate)for(int op=0;op<4;++op)require(std::memcmp(values[op].contents,reference[variant][op].data(),reference[variant][op].size()*4)==0,"MLP intermediate, output or padding changed bits");++executions;
    return std::array<double,3>{ms(begin,encoded),ms(begin,end),(command.GPUEndTime-command.GPUStartTime)*1000};
   }};
   double worstProjection=0,worstSilu=0;
   for(int variant=0;variant<4;++variant){run(0,variant,false);for(int op=0;op<4;++op){reference[variant][op].resize(rows*(op==3?hStride:iStride));std::memcpy(reference[variant][op].data(),values[op].contents,reference[variant][op].size()*4);}
    for(int op=0;op<4;++op)for(uint64_t row=0;row<rows;++row){const uint64_t columns=op==3?hidden:intermediate,stride=op==3?hStride:iStride;for(uint64_t col=0;col<stride;++col){const double value=reference[variant][op][row*stride+col];require(std::isfinite(value),"Nonfinite MLP output");if(col>=columns){require(value==99.5,"MLP padding overwritten");continue;}
      if(op==2){const double gate=reference[variant][0][row*iStride+col],up=reference[variant][1][row*iStride+col],expected=(gate/(1+std::exp(-gate)))*up;worstSilu=std::max(worstSilu,std::abs(value-expected)/(5e-6+5e-5*std::abs(expected)));}
      else {const uint64_t inner=op==3?intermediate:hidden;const auto rawWeight=static_cast<const uint16_t*>(raw[op==3?2:op].bytes);double sum=0,absolute=0;
       for(uint64_t k=0;k<inner;++k){uint32_t bits=uint32_t(rawWeight[col*inner+k])<<16;float w;std::memcpy(&w,&bits,4);const double x=op==3?reference[variant][2][row*iStride+k]:base[row*hStride+k]*scales[variant];const double product=double(w)*x;sum+=product;absolute+=std::abs(product);}
       const double epsilon=std::ldexp(1.0,-23),steps=std::ceil(double(inner)/32)+32,gamma=steps*epsilon/(1-steps*epsilon);worstProjection=std::max(worstProjection,std::abs(value-sum)/(gamma*absolute+1e-7));
      }
    }}
   }require(worstProjection<=1&&worstSilu<=1,"MLP independent numerical bound exceeded");
   for(int pair=0;pair<3;++pair)for(int round=0;round<2;++round)for(bool candidate:{false,true,true,false}){const int mode=candidate?(pair==1?3:2):(pair==2?1:0);
    for(int warm=0;warm<3;++warm)run(mode,int(epoch++%4),true);auto encoding=[NSMutableArray new],wall=[NSMutableArray new],gpu=[NSMutableArray new];
    for(int sample=0;sample<12;++sample){auto t=run(mode,int(epoch++%4),true);[encoding addObject:@(t[0])];[wall addObject:@(t[1])];[gpu addObject:@(t[2])];}
    emit(@{@"kind":@"timing",@"model":model,@"rows":@(rows),@"pair":@(pair),@"round":@(round),@"policy":candidate?@"candidate":@"control",@"encode_ms":encoding,@"wall_ms":wall,@"gpu_ms":gpu,@"bits_exact":@YES});
   }
   emit(@{@"kind":@"check",@"model":model,@"rows":@(rows),@"bits_exact":@YES,@"padding_preserved":@YES,@"input_variants":@4,@"projection_fp64_worst_ratio":@(worstProjection),@"silu_worst_ratio":@(worstSilu),@"validated_executions":@360});
  }}
 }}emit(@{@"kind":@"completed",@"profiles":@6,@"executions":@(executions)});return 0;
}catch(const std::exception& error){std::fprintf(stderr,"indirect_mlp_failed: %s\n",error.what());return 1;}}}
