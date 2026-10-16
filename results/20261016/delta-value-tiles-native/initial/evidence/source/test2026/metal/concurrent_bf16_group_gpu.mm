// Independent learned projections: separate encoders, one serial pass, one concurrent pass.
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
struct P{uint64_t rows,inner,columns,aStride,oStride;};
struct Projection{id<MTLBuffer> weight,out;P p;std::vector<float> reference[4];};
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static void emit(NSDictionary* x){auto raw=[NSJSONSerialization dataWithJSONObject:x options:0 error:nil];need(raw!=nil,"JSON failed");puts([[NSString alloc] initWithData:raw encoding:NSUTF8StringEncoding].UTF8String);fflush(stdout);}
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==3,"Expected original shader and grouped assets");auto d=MTLCreateSystemDefaultDevice();auto queue=[d newCommandQueue];need(d&&queue,"Metal unavailable");NSError* error=nil;
 auto source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
 auto library=[d newLibraryWithSource:source options:nil error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto pipeline=[&](NSString* name){auto p=[d newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);need(p.threadExecutionWidth==32,"SIMD width changed");return p;};
 id<MTLComputePipelineState> kernels[]={pipeline(@"bf16_decode"),pipeline(@"bf16_decode_small")};
 auto folder=[NSString stringWithUTF8String:argv[2]];auto manifest=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"groups.json"]] options:0 error:nil];need(manifest!=nil,"Group manifest unavailable");
 emit(@{@"kind":@"hardware",@"device":d.name,@"shader_math":@"Original native BF16/default options; FP32 input, arithmetic and output"});
 for(NSDictionary* group in manifest[@"groups"]){for(uint64_t rows:{1,2}){@autoreleasepool{
  NSString* name=group[@"name"];NSArray* cases=group[@"cases"];need(cases.count>=2&&cases.count<=3,"Expected two or three independent projections");
  uint64_t k=[cases[0][@"inner"] unsignedLongLongValue],aStride=((k+15)/16)*16+16;
  std::vector<float> base(rows*aStride,0);for(uint64_t r=0;r<rows;++r)for(uint64_t j=0;j<k;++j){float x=std::sin(float(r*k+j)*.013f)*.3f;if(j%211==0)x=-0.0f;base[r*aStride+j]=x;}
  auto input=[d newBufferWithLength:base.size()*4 options:MTLResourceStorageModeShared];need(input!=nil,"Input allocation failed");std::vector<Projection> projections;double fp64Worst=0;
  for(NSDictionary* c in cases){uint64_t n=[c[@"columns"] unsignedLongLongValue];need([c[@"inner"] unsignedLongLongValue]==k,"Independent projections must share the same input width");
   auto raw=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:[c[@"name"] stringByAppendingString:@".bin"]]];need(raw.length==k*n*2,"Weight byte mismatch");
   Projection v;v.p={rows,k,n,aStride,((n+15)/16)*16+16};v.weight=[d newBufferWithBytes:raw.bytes length:raw.length options:MTLResourceStorageModeShared];v.out=[d newBufferWithLength:rows*v.p.oStride*4 options:MTLResourceStorageModeShared];need(v.weight&&v.out,"Projection allocation failed");projections.push_back(std::move(v));
  }
  const float scales[]={1,-1,.5f,2};size_t validated=0,epoch=0;
  auto run=[&](int mode,int variant,bool validate){
   for(size_t i=0;i<base.size();++i)static_cast<float*>(input.contents)[i]=base[i]*scales[variant];
   for(auto& v:projections)std::fill_n(static_cast<float*>(v.out.contents),rows*v.p.oStride,99.5f);
   auto begin=Clock::now();auto cb=[queue commandBuffer];need(cb!=nil,"Command unavailable");
   id<MTLComputeCommandEncoder> encoder=nil;
   if(mode)encoder=[cb computeCommandEncoderWithDispatchType:mode==2?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial];
   for(auto& v:projections){if(mode==0)encoder=[cb computeCommandEncoder];need(encoder!=nil,"Compute encoder unavailable");bool small=v.p.columns<=256;
    [encoder setComputePipelineState:kernels[small]];[encoder setBuffer:input offset:0 atIndex:0];[encoder setBuffer:v.weight offset:0 atIndex:1];[encoder setBuffer:v.out offset:0 atIndex:2];[encoder setBytes:&v.p length:sizeof(v.p) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake((v.p.columns+(small?3:15))/(small?4:16),1,1) threadsPerThreadgroup:MTLSizeMake(small?64:128,1,1)];if(mode==0)[encoder endEncoding];
   }
   if(mode)[encoder endEncoding];auto encoded=Clock::now();[cb commit];[cb waitUntilCompleted];auto end=Clock::now();need(cb.status==MTLCommandBufferStatusCompleted,"Grouped projection command failed");
   if(validate){for(auto& v:projections)need(std::memcmp(v.out.contents,v.reference[variant].data(),rows*v.p.oStride*4)==0,"Projection or padding changed bits");++validated;}
   return std::array<double,3>{ms(begin,encoded),ms(begin,end),(cb.GPUEndTime-cb.GPUStartTime)*1000};
  };
  for(int variant=0;variant<4;++variant){run(0,variant,false);for(auto& v:projections){v.reference[variant].resize(rows*v.p.oStride);std::memcpy(v.reference[variant].data(),v.out.contents,rows*v.p.oStride*4);}}
  for(auto& v:projections){auto weights=static_cast<const uint16_t*>(v.weight.contents);double epsilon=std::ldexp(1.0,-23),operations=std::ceil(double(k)/32)+32,gamma=operations*epsilon/(1-operations*epsilon);
   for(uint64_t r=0;r<rows;++r)for(uint64_t col=0;col<v.p.columns;++col){double exact=0,sumabs=0;
    for(uint64_t j=0;j<k;++j){uint32_t bits=uint32_t(weights[col*k+j])<<16;float w;std::memcpy(&w,&bits,4);double z=double(base[r*aStride+j])*w;exact+=z;sumabs+=std::abs(z);}
    for(int variant=0;variant<4;++variant){double x=v.reference[variant][r*v.p.oStride+col],bound=gamma*sumabs*std::abs(scales[variant])+1e-7;need(std::isfinite(x)&&std::isfinite(bound),"Nonfinite reference");fp64Worst=std::max(fp64Worst,std::abs(x-exact*scales[variant])/bound);}
   }
   for(int variant=0;variant<4;++variant)for(uint64_t r=0;r<rows;++r)for(uint64_t col=v.p.columns;col<v.p.oStride;++col)need(v.reference[variant][r*v.p.oStride+col]==99.5f,"Control changed output padding");
  }need(fp64Worst<=1,"Original FP64 envelope exceeded");
  for(int mode=0;mode<3;++mode)for(int variant=0;variant<4;++variant)run(mode,variant,true);
  auto warm=Clock::now();do{int mode=int(epoch%3),variant=int(epoch%4);++epoch;run(mode,variant,true);}while(ms(warm,Clock::now())<250);
  emit(@{@"kind":@"warmup",@"group":name,@"rows":@(rows),@"elapsed_ms":@(ms(warm,Clock::now()))});
  for(int candidate:{1,2})for(int round=0;round<2;++round){int window=0;for(bool isCandidate:{false,true,true,false}){
   int mode=isCandidate?candidate:0;for(int i=0;i<3;++i)run(mode,int(epoch++%4),true);auto enc=[NSMutableArray new],wall=[NSMutableArray new],gpu=[NSMutableArray new];
   for(int sample=0;sample<12;++sample){auto t=run(mode,int(epoch++%4),true);[enc addObject:@(t[0])];[wall addObject:@(t[1])];[gpu addObject:@(t[2])];}
   emit(@{@"kind":@"timing",@"group":name,@"rows":@(rows),@"candidate_mode":@(candidate),@"round":@(round),@"window":@(window++),@"mode":@(mode),@"encode_ms":enc,@"wall_ms":wall,@"gpu_ms":gpu});
  }}
  emit(@{@"kind":@"check",@"group":name,@"rows":@(rows),@"projections":@(projections.size()),@"input_variants":@4,@"bits_exact":@YES,@"padding_preserved":@YES,@"fp64_worst_ratio":@(fp64Worst),@"validated_executions":@(validated)});
 }}}
 return 0;
}catch(const std::exception& e){fprintf(stderr,"concurrent_bf16_group_failed: %s\n",e.what());return 1;}}}
