// Current expanded FP32 MPS vision products versus exact common half-storage alpha.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "../../src/backends/metal/exact_half_codec.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using Clock=std::chrono::steady_clock;
struct Matrix {id<MTLBuffer> buffer;MPSMatrix* view;uint64_t stride;};
struct Params {uint64_t rows,columns,iStride,oStride;};
static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static void emit(NSDictionary* value){auto d=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];need(d!=nil,"JSON failed");puts([[NSString alloc] initWithData:d encoding:NSUTF8StringEncoding].UTF8String);fflush(stdout);}
static Matrix matrix(id<MTLDevice> d,size_t rows,size_t cols,MPSDataType type){
 auto bytes=[MPSMatrixDescriptor rowBytesFromColumns:cols dataType:type];auto descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:bytes dataType:type];
 auto buffer=[d newBufferWithLength:rows*bytes options:MTLResourceStorageModeShared];need(buffer!=nil,"Matrix allocation failed");std::memset(buffer.contents,0,rows*bytes);
 return {buffer,[[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor],bytes/(type==MPSDataTypeFloat16?2:4)};
}
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==2,"Expected selected matrix manifest");NSError* error=nil;auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];
 auto manifest=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]] options:0 error:nil];size_t total=0;
 for(NSDictionary* item in manifest[@"cases"]){@autoreleasepool{
  auto bytes=[NSData dataWithContentsOfFile:item[@"path"]];const uint64_t inner=[item[@"inner"] unsignedLongLongValue],columns=[item[@"columns"] unsignedLongLongValue];need(bytes.length==inner*columns*2,"BF16 byte count mismatch");
  const auto raw=static_cast<const uint16_t*>(bytes.bytes);uint16_t maximum=0;for(size_t i=0;i<inner*columns;++i)maximum=std::max(maximum,uint16_t(raw[i]&32767));
  int32_t globalScale=0;bool eligible=Garnet::ExactHalfRowScale(maximum,globalScale);std::vector<uint16_t> globalBits(inner*columns);std::vector<float> learned(inner*columns);
  for(size_t i=0;i<inner*columns&&eligible;++i){eligible=Garnet::EncodeExactScaledHalf(raw[i],globalScale,globalBits[i]);uint32_t bits=uint32_t(raw[i])<<16;std::memcpy(&learned[i],&bits,4);}
  if(!eligible){emit(@{@"kind":@"excluded",@"model":item[@"model"],@"tensor":item[@"tensor"],@"reason":@"Common scale is not exact"});continue;}
  auto floatWeight=matrix(device,columns,inner,MPSDataTypeFloat32),globalWeight=matrix(device,columns,inner,MPSDataTypeFloat16);
  for(uint64_t col=0;col<columns;++col)for(uint64_t k=0;k<inner;++k){
   static_cast<float*>(floatWeight.buffer.contents)[col*floatWeight.stride+k]=learned[col*inner+k];
   static_cast<uint16_t*>(globalWeight.buffer.contents)[col*globalWeight.stride+k]=globalBits[col*inner+k];
   __fp16 half;std::memcpy(&half,&globalBits[col*inner+k],2);float restored=std::ldexp(float(half),-globalScale);
   need(std::memcmp(&restored,&learned[col*inner+k],4)==0,"Vision half storage changed learned bits");
  }
  for(NSNumber* rowValue in item[@"rows"]){@autoreleasepool{
   const uint64_t rows=rowValue.unsignedLongLongValue;auto input=matrix(device,rows,inner,MPSDataTypeFloat32),output=matrix(device,rows,columns,MPSDataTypeFloat32);
   std::vector<float> base(rows*input.stride,0);for(uint64_t r=0;r<rows;++r)for(uint64_t k=0;k<inner;++k)base[r*input.stride+k]=std::sin(float(r*inner+k)*.013f)*.3f;
   auto control=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:columns interiorColumns:inner alpha:1 beta:0];
   auto candidate=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:columns interiorColumns:inner alpha:std::ldexp(1.0,-globalScale) beta:0];
   const float variants[4]={1,-1,.5f,2};std::vector<float> reference[4];size_t epoch=0,validated=0;double maxAbs=0,comparisonRatio=0,controlRatio=0,candidateRatio=0;
   auto run=[&](bool useCandidate,int variant,bool validate){@autoreleasepool{
    for(size_t i=0;i<base.size();++i)static_cast<float*>(input.buffer.contents)[i]=base[i]*variants[variant];std::fill_n(static_cast<float*>(output.buffer.contents),rows*output.stride,99.5f);
    auto start=Clock::now();auto command=[queue commandBuffer];
    [(useCandidate?candidate:control) encodeToCommandBuffer:command leftMatrix:input.view rightMatrix:(useCandidate?globalWeight:floatWeight).view resultMatrix:output.view];
    auto encoded=Clock::now();[command commit];[command waitUntilCompleted];auto end=Clock::now();need(command.status==MTLCommandBufferStatusCompleted,"MPS GPU command failed");
    if(validate){auto actual=static_cast<const float*>(output.buffer.contents);for(uint64_t r=0;r<rows;++r)for(uint64_t c=0;c<output.stride;++c){auto i=r*output.stride+c;need(std::isfinite(actual[i]),"Nonfinite MPS projection");if(c>=columns){need(actual[i]==99.5f,"Output padding overwritten");continue;}
      double difference=std::abs(double(actual[i])-reference[variant][i]);maxAbs=std::max(maxAbs,difference);comparisonRatio=std::max(comparisonRatio,difference/(5e-6+5e-5*std::abs(double(reference[variant][i]))));}
     need(comparisonRatio<=1,"Global alpha exceeds original comparison envelope");++validated;}
    ++total;return std::array<double,3>{ms(start,encoded),ms(start,end),(command.GPUEndTime-command.GPUStartTime)*1000};
   }};
   // Full independent FP64 projection, reused by exact power-of-two input variants.
   std::vector<double> exact(rows*columns);for(uint64_t r=0;r<rows;++r)for(uint64_t c=0;c<columns;++c){double sum=0;for(uint64_t k=0;k<inner;++k)sum+=double(base[r*input.stride+k])*learned[c*inner+k];exact[r*columns+c]=sum;}
   for(int v=0;v<4;++v){run(false,v,false);reference[v].resize(rows*output.stride);std::memcpy(reference[v].data(),output.buffer.contents,reference[v].size()*4);
    for(uint64_t r=0;r<rows;++r)for(uint64_t c=0;c<columns;++c){double expected=exact[r*columns+c]*variants[v];controlRatio=std::max(controlRatio,std::abs(double(reference[v][r*output.stride+c])-expected)/(5e-6+5e-5*std::abs(expected)));}
    run(true,v,true);auto actual=static_cast<const float*>(output.buffer.contents);for(uint64_t r=0;r<rows;++r)for(uint64_t c=0;c<columns;++c){double expected=exact[r*columns+c]*variants[v];candidateRatio=std::max(candidateRatio,std::abs(double(actual[r*output.stride+c])-expected)/(5e-6+5e-5*std::abs(expected)));}
   }need(controlRatio<=1&&candidateRatio<=1,"MPS projection exceeds unchanged independent FP64 envelope");
   auto warm=Clock::now();do{run(false,int(epoch%4),true);run(true,int(epoch++%4),true);}while(ms(warm,Clock::now())<250);double warmMs=ms(warm,Clock::now());
   for(int round=0;round<2;++round)for(bool policy:{false,true,true,false}){for(int i=0;i<3;++i)run(policy,int(epoch++%4),true);auto encoding=[NSMutableArray new],wall=[NSMutableArray new],gpu=[NSMutableArray new];
    for(int i=0;i<12;++i){auto t=run(policy,int(epoch++%4),true);[encoding addObject:@(t[0])];[wall addObject:@(t[1])];[gpu addObject:@(t[2])];}
    emit(@{@"kind":@"timing",@"model":item[@"model"],@"tensor":item[@"tensor"],@"rows":@(rows),@"round":@(round),@"candidate":@(policy),@"encode_ms":encoding,@"wall_ms":wall,@"gpu_ms":gpu});
   }
   emit(@{@"kind":@"check",@"model":item[@"model"],@"tensor":item[@"tensor"],@"rows":@(rows),@"inner":@(inner),@"columns":@(columns),@"global_scale":@(globalScale),@"learned_bits_exact":@YES,@"padding_preserved":@YES,@"input_variants":@4,@"max_abs_error":@(maxAbs),@"comparison_worst_ratio":@(comparisonRatio),@"control_fp64_worst_ratio":@(controlRatio),@"candidate_fp64_worst_ratio":@(candidateRatio),@"validated_executions":@(validated),@"common_warm_ms":@(warmMs)});
  }}
 }}emit(@{@"kind":@"completed",@"executions":@(total)});return 0;
}catch(const std::exception& e){fprintf(stderr,"vision_global_alpha_failed: %s\n",e.what());return 1;}}}
