// Original add + RMS operations versus one dispatch writing both outputs.
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
struct Params {uint64_t rows,columns,aStride,bStride,cStride;float epsilon,offset;};
static void need(bool x,const char* s){if(!x)throw std::runtime_error(s);}
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static void emit(NSDictionary* x){auto d=[NSJSONSerialization dataWithJSONObject:x options:0 error:nil];need(d!=nil,"JSON failure");puts([[NSString alloc] initWithData:d encoding:NSUTF8StringEncoding].UTF8String);fflush(stdout);}
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==3,"Expected shader and norm manifest");NSError* error=nil;
 auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];
 auto source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
 auto library=[device newLibraryWithSource:source options:nil error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 id<MTLComputePipelineState> pipe[3];NSString* names[3]={@"add_matrix",@"rms_norm",@"residual_norm"};
 for(int i=0;i<3;++i){pipe[i]=[device newComputePipelineStateWithFunction:[library newFunctionWithName:names[i]] error:&error];need(pipe[i]!=nil,"Pipeline failure");}
 auto manifest=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]] options:0 error:nil];
 size_t executions=0;
 for(NSDictionary* item in manifest[@"cases"]){@autoreleasepool{
  uint64_t columns=[item[@"columns"] unsignedLongLongValue];float offset=[item[@"offset"] floatValue],eps=[item[@"epsilon"] floatValue];
  auto raw=[NSData dataWithContentsOfFile:item[@"path"]];need(raw.length==columns*4,"Norm data size mismatch");
  auto weight=[device newBufferWithBytes:raw.bytes length:raw.length options:MTLResourceStorageModeShared];
  for(uint64_t rows:{1,2,16,32,64}){@autoreleasepool{
   uint64_t stride=((columns+15)/16)*16+16;Params add{rows,columns,stride,stride,stride,0,0},norm{rows,columns,stride,columns,stride,eps,offset};
   std::vector<float> a(rows*stride),b(rows*stride);for(uint64_t r=0;r<rows;++r)for(uint64_t k=0;k<columns;++k){a[r*stride+k]=std::sin(float(r*columns+k)*.013f)*.3f;b[r*stride+k]=std::cos(float(r*columns+k)*.027f)*.2f;}
   auto ab=[device newBufferWithLength:a.size()*4 options:MTLResourceStorageModeShared],bb=[device newBufferWithLength:b.size()*4 options:MTLResourceStorageModeShared];
   auto residual=[device newBufferWithLength:a.size()*4 options:MTLResourceStorageModeShared],out=[device newBufferWithLength:a.size()*4 options:MTLResourceStorageModeShared];
   std::vector<float> ref[4][2];const float scales[4]={1,-1,.5f,2};size_t epoch=0,validated=0;
   auto run=[&](bool fused,int variant,bool validate){@autoreleasepool{
    for(size_t i=0;i<a.size();++i){static_cast<float*>(ab.contents)[i]=a[i]*scales[variant];static_cast<float*>(bb.contents)[i]=b[i]*scales[(variant+1)%4];}
    std::fill_n(static_cast<float*>(residual.contents),a.size(),99.5f);std::fill_n(static_cast<float*>(out.contents),a.size(),99.5f);
    auto begin=Clock::now();auto command=[queue commandBuffer];
    if(fused){auto e=[command computeCommandEncoder];[e setComputePipelineState:pipe[2]];[e setBuffer:ab offset:0 atIndex:0];[e setBuffer:bb offset:0 atIndex:1];[e setBuffer:weight offset:0 atIndex:2];[e setBuffer:residual offset:0 atIndex:3];[e setBuffer:out offset:0 atIndex:4];[e setBytes:&norm length:sizeof(norm) atIndex:5];[e dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}
    else {auto e=[command computeCommandEncoder];[e setComputePipelineState:pipe[0]];[e setBuffer:ab offset:0 atIndex:0];[e setBuffer:bb offset:0 atIndex:1];[e setBuffer:residual offset:0 atIndex:2];[e setBytes:&add length:sizeof(add) atIndex:3];[e dispatchThreads:MTLSizeMake(rows*columns,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(uint64_t(pipe[0].maxTotalThreadsPerThreadgroup),rows*columns),1,1)];[e endEncoding];
     e=[command computeCommandEncoder];[e setComputePipelineState:pipe[1]];[e setBuffer:residual offset:0 atIndex:0];[e setBuffer:weight offset:0 atIndex:1];[e setBuffer:out offset:0 atIndex:2];[e setBytes:&norm length:sizeof(norm) atIndex:3];[e dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}
    auto encoded=Clock::now();[command commit];[command waitUntilCompleted];auto end=Clock::now();need(command.status==MTLCommandBufferStatusCompleted,"GPU command failure");
    if(validate){need(std::memcmp(residual.contents,ref[variant][0].data(),a.size()*4)==0,"Residual bits or padding changed");need(std::memcmp(out.contents,ref[variant][1].data(),a.size()*4)==0,"Norm bits or padding changed");++validated;}++executions;
    return std::array<double,3>{ms(begin,encoded),ms(begin,end),(command.GPUEndTime-command.GPUStartTime)*1000};
   }};
   double worst=0;
   for(int v=0;v<4;++v){run(false,v,false);for(int op=0;op<2;++op){ref[v][op].resize(a.size());std::memcpy(ref[v][op].data(),op?out.contents:residual.contents,a.size()*4);}
    for(uint64_t r=0;r<rows;++r){double sum=0;for(uint64_t k=0;k<columns;++k){float expected=a[r*stride+k]*scales[v]+b[r*stride+k]*scales[(v+1)%4];need(ref[v][0][r*stride+k]==expected,"Independent residual mismatch");sum+=double(expected)*expected;}
     double inverse=1/std::sqrt(sum/columns+eps);for(uint64_t k=0;k<columns;++k){double x=double(ref[v][0][r*stride+k])*inverse*(double(static_cast<const float*>(raw.bytes)[k])+offset);double actual=ref[v][1][r*stride+k];need(std::isfinite(actual)&&std::isfinite(x),"Nonfinite norm");worst=std::max(worst,std::abs(actual-x)/(5e-6+5e-5*std::abs(x)));}
     for(uint64_t k=columns;k<stride;++k)need(ref[v][0][r*stride+k]==99.5f&&ref[v][1][r*stride+k]==99.5f,"Control padding changed");}
    run(true,v,true);
   }need(worst<=1,"Independent FP64 norm bound exceeded");
   auto warm=Clock::now();do{run(false,int(epoch%4),true);run(true,int(epoch++%4),true);}while(ms(warm,Clock::now())<250);
   for(int round=0;round<3;++round)for(bool fused:{false,true,true,false}){for(int i=0;i<3;++i)run(fused,int(epoch++%4),true);auto enc=[NSMutableArray new],wall=[NSMutableArray new],gpu=[NSMutableArray new];
    for(int sample=0;sample<24;++sample){auto t=run(fused,int(epoch++%4),true);[enc addObject:@(t[0])];[wall addObject:@(t[1])];[gpu addObject:@(t[2])];}
    emit(@{@"kind":@"timing",@"model":item[@"model"],@"tensor":item[@"tensor"],@"rows":@(rows),@"round":@(round),@"policy":fused?@"candidate":@"control",@"encode_ms":enc,@"wall_ms":wall,@"gpu_ms":gpu,@"bits_exact":@YES});
   }
   emit(@{@"kind":@"check",@"model":item[@"model"],@"tensor":item[@"tensor"],@"rows":@(rows),@"columns":@(columns),@"bits_exact":@YES,@"padding_preserved":@YES,@"input_variants":@4,@"fp64_worst_ratio":@(worst),@"validated_executions":@(validated),@"common_warm_ms":@(ms(warm,Clock::now()))});
  }}
 }}emit(@{@"kind":@"completed",@"executions":@(executions)});return 0;
}catch(const std::exception& e){fprintf(stderr,"residual_norm_failed: %s\n",e.what());return 1;}}}
